function [ Pact, Vtracker ] = ModelBayes( beta, gamma, fb, mags, GenerateConditionals )
% returns the bayes model's actions given beta and gamma. beta and gamma must be column vectors of the same length; we vectorize over them.
% fb is the actual bottom-level samples seen by the LSTM. 

nP = length(beta);  % number of parameter points to vectorize over
nTr = size(fb,1);   % number of trials per episode
nEp = size(fb,2);  % number of episodes

% note we'll keep the dimensions of everything as r(i+1), v(i+1), r(i), v(i), k 
[nk, nv, nr, lv, lr, v_sup, r_sup] = GenerateConditionals();  % call that function handle, which already has kMin, kMax and vMin, vMax if we need them


Pact = nan(nTr, nEp, nP);
Vtracker = nan(nTr, nEp);

%% debug code
% keyboard  % what does the posterior look like after a high-vol-first half of an episode? why can't the vol quickly come back down?
% vObj = VideoWriter('testVid.avi', 'Motion JPEG AVI'); vObj.FrameRate = 6; vObj.Quality = 100; open(vObj);
% ftc = figure(1); set(ftc, 'Position', [0 0 1800 600])


for iEp=1:nEp
    if mod(iEp,100)==0, disp(['simulating actions for bayes model, ep ' num2str(iEp) ' of ' num2str(nEp)]); end
    
    %% set the prior to uniform for each episode
    p = ones(nr, nv, 1, 1, nk)/(nr*nv*nk); %  r(i+1), v(i+1), _, _, k

    for iTr=1:nTr-1
        %% probability of action on this trial
        p_r = sum(sum(p,5),2); % marginal probability of r (integrating over v(i+1) and k)   [what we're calling v(i+1) here, from the end of the last trial, pertains to the start of this trial]
        V = sum(p_r' .* r_sup);  % expectation of r

        %% apply risk preference
        F0 = max(min(gamma.*((1-V)-0.5)+0.5,1),0);
        F1 = max(min(gamma.*(V-0.5)+0.5,1),0);
        
        %% make choice
        G0 = F0 * mags(iTr, iEp, 1);
        G1 = F1 * mags(iTr, iEp, 2);
        Pact(iTr,iEp,:) = 1 ./ (1 + exp(-beta.*(G1-G0)));
        
        %% track believed volatility
        Vtracker(iTr,iEp) = sum(sum(p,1),5) * v_sup';
        
        %% update posterior for next trial
        post = permute(p, [3 4 1 2 5]); % last trial's v(i+1) is this trial's v(i), and last trial's r(i+1) is this trial's r(i)
        post = sum(post .* lv, 4);  % apply model of meta-volatility, and integrate over v(i)
        post = sum(post .* lr, 3); % apply model of volatility, and integrate over r(i)
        ly = fb(iTr,iEp) * r_sup' + (1-fb(iTr,iEp)) * (1-r_sup');  % conditional probability of getting the outcome we saw, given r(i+1)
        post = post .* ly;
        p = post / sum(post(:)); % re-normalize
        
        %% debug plotting
%         figure(1), pltinds = [1 2 3 4 7 8 9 10 13 14 15];
%         for ik=1:nk
%             subplot(3,6,pltinds(ik))
%             imagesc(v_sup, r_sup, p(:,:,1,1,ik)), ylabel('r'), xlabel('v'), title(['ik=' num2str(ik)]), caxis([0 8e-3])
%         end
%         subplot(3,6,[5 6 11 12]), plot(fb(1:iTr-1,iEp), 'bo'), hold on, plot(iTr, fb(iTr,iEp), 'ro'), xlim([0 200])
%         subplot(3,6,16), plot(squeeze(sum(sum(p,1),2))), xlabel('ik'), ylabel('p(ik)')
%         subplot(3,6,17), plot(v_sup, squeeze(sum(sum(p,1),5))), xlabel('v'), ylabel('p(v)')
%         subplot(3,6,18), plot(r_sup, squeeze(sum(sum(p,2),5))), xlabel('r'), ylabel('p(r)')
%         frame = getframe(ftc);
%         writeVideo(vObj,frame);
    end
end

%% debug
% close(vObj);



end


