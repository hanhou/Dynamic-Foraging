
clear all hidden, close all

%% conventions
% 1) baseline at time t causes action at time t which causes reward at time t:
%      r(t-1)   ...->    v(t) -> a(t) -> r(t)    ...->    v(t+1)
%    v(1) is arbitrarily set, and we don't know v(epLength+1).
%    we crop off the first step of "reward" numbers from the LSTM's output, to align with this convention. 
%    we're missing r(end) because it doesn't get written by the agent code.
% 2) we name the two actions 0 and 1.
% 3) in matrices with trial and episode dimensions, trial dimension comes first, like: nTr*nEp
% 4) action probabilities are of chosing action_1 (versus action_0)
% 5) the 'PR' values are probabilities that 1 is correct (versus 0)
% 6) mags(:,:,1) corresponds to action0, and mags(:,:,2) corresponds to action1

%% parameters for R-W model
maxAlpha = 1;  % the maximum learning rate to evaluate for rescorla-wagner models
windowSize = 10; % sliding window size (in units of trials) for estimating changing learning rates
num_alphas = 101;  % grid size for learning rates
minlogbeta = -1;
maxlogbeta = 2;
num_betas = 21;

%% parameters for both R-W and bayesian model
simBeta = 10000; % softmax inverse temperature to generate choice data from models

%% parameters for bayesian model
simGamma = 1;  % risk preference parameter from Behrens et al 2007, for generating choice data
kMin = -4.5; kMax = -3.5; % allowed range of meta-volatility
vMin = 0; vMax = 0.2;  % allowed range of volatility

run_local_path = 'Data';

%% extract data from .mat into convenient variables
mat_files = dir([run_local_path '/*.mat']); mat_files = {mat_files(:).name}; nEp = length(mat_files); 
S = load([run_local_path '/' mat_files{1}]); 
nTr = size(S.action_history,2); 
nHidden = size(S.hiddens,3);
rew_LSTM = nan(size(S.reward_history,2),nEp);
PR = nan(size(S.probs_history,2),nEp);
hiddens = nan(size(S.hiddens,2), nHidden, nEp); 
baselines = nan(size(S.baselines,2),nEp);
act_LSTM = nan(size(S.action_history,2), nEp);
mags = nan(size(S.mag_history,1), nEp, 2);
for iEp=1:nEp
    S = load([run_local_path '/' mat_files{iEp}]);
    PR(:,iEp) = S.probs_history;
    rew_LSTM(:,iEp) = S.reward_history;
    hiddens(:,:,iEp) = S.hiddens;
    baselines(:,iEp) = S.baselines;
    act_LSTM(:,iEp) = S.action_history;
    mags(:,iEp,:) = S.mag_history;
end
clear S
rew_LSTM = rew_LSTM(2:end,:);   % shift rewards ahead by one timestep, because the files record the reward for trial i in row i+1
fb = double((rew_LSTM>0)==act_LSTM); 
fb(end,:) = nan; 
HVF = abs(PR(1,:) - 0.5) > 0.275;  % the episodes with High Volatility First
clear rew_LSTM;  % don't be tempted to use this variable. we'll stick to the convention of calculating rewards from the "feedback"

%% simulate other agents
alphas = linspace(0,maxAlpha,num_alphas); 
nAp = length(alphas);
Pact_RW = permute(ModelRW(fb, mags, alphas', repmat(simBeta, [nAp 1])), [2 3 1]);
Pact_Bayes = ModelBayes(simBeta, simGamma, fb, mags, @() GenerateConditionals_Reversal(kMin, kMax, vMin, vMax));

%% sample concrete actions based on the action probabilities
act_Bayes = rand(nTr,nEp) < Pact_Bayes;

%% fit the overall R-W model, to get probs that we'll use for both single-alpha and many-alpha models.
betas = logspace(minlogbeta,maxlogbeta,num_betas);
[px, py] = ndgrid(alphas, betas);
alphas_grid = px(:);
betas_grid = py(:);
mprobs = ModelRW(fb, mags, alphas_grid, betas_grid);

%% fit each behavior to R-W single-alpha model
mlikes = LogLike(act_LSTM, mprobs); 
nLL_LSTM_a1 = min(mlikes);
mlikes = LogLike(act_Bayes, mprobs); 
nLL_Bayes_a1 = min(mlikes);

%% fit each behavior to R-W many-alpha model
fmuopts = optimoptions('fminunc', 'Display', 'off', 'Algorithm', 'quasi-newton', 'MaxFunctionEvaluations', 1e7); 
fitAlpha_LSTM = nan(nTr, 2);  % 2 columns of fitAlpha are: high-vol-first episodes, low-vol-first episodes
nLL_LSTM_aM = nan(nTr,1);   
fitAlpha_Bayes = nan(nTr, 2); 
nLL_Bayes_aM = nan(nTr,1);
for iSW = 1:(nTr-windowSize+1)  % slide the sliding window
    disp(['fitting behavior in sliding window: ' num2str(iSW) '/' num2str(nTr-windowSize+1)])
    trialMask = false(nTr,1); 
    trialMask(iSW:(iSW+windowSize-1)) = true;
    
    %% LSTM behavior
    % high volatility first
    mlikes1 = LogLike(act_LSTM(trialMask,HVF), mprobs(:,trialMask,HVF));
    [ll1,ix] = min(mlikes1); 
    [tmp, ~] = ind2sub([num_alphas num_betas], ix); 
    fitAlpha_LSTM(iSW,1) = alphas_grid(tmp);
    % low volatility first
    mlikes2 = LogLike(act_LSTM(trialMask,~HVF), mprobs(:,trialMask,~HVF));
    [ll2,ix] = min(mlikes2); 
    [tmp, ~] = ind2sub([num_alphas num_betas], ix); 
    fitAlpha_LSTM(iSW,2) = alphas_grid(tmp);
    nLL_LSTM_aM(iSW) = ll1+ll2;
    
    %% Bayes behavior
    mlikes1 = LogLike(act_Bayes(trialMask,HVF), mprobs(:,trialMask,HVF));
    [ll1,ix] = min(mlikes1); 
    [tmp, ~] = ind2sub([num_alphas num_betas], ix); 
    fitAlpha_Bayes(iSW,1) = alphas_grid(tmp);
    mlikes2 = LogLike(act_Bayes(trialMask,~HVF), mprobs(:,trialMask,~HVF));
    [ll2,ix] = min(mlikes2); 
    [tmp, ~] = ind2sub([num_alphas num_betas], ix); 
    fitAlpha_Bayes(iSW,2) = alphas_grid(tmp);
    nLL_Bayes_aM(iSW) = ll1+ll2;
end

%% fit each behavior to Bayes model with risk params
[ params_LSTM_Bayes, nLL_LSTM_Bayes, Vtracker ] = FitBayes( act_LSTM, fb, mags, kMin, kMax, vMin, vMax, 'reversal' );

%% estimate a cross-validated decoding of the volatility signal.
nCV = 100; 
hPerm = permute(hiddens(1:(nTr-1),:,:), [2 1 3]);
decoded_volatility = nan(nCV, nTr-1, nEp);
for iCV = 1:nCV
    disp(['decoding volatility: ' num2str(iCV) '/' num2str(nCV)])
    iLO = randsample(1:nEp, floor(0.8*nEp));  % leave out 80% of episodes
    iLI = setdiff(1:nEp, iLO); 
    B = glmfit(reshape(hPerm(:,:,iLI), [nHidden (nTr-1)*length(iLI)])', reshape(Vtracker(1:(nTr-1),iLI), [(nTr-1)*length(iLI) 1]));  % predict bayes volatility signal
    X = reshape(hPerm(:,:,iLO), [nHidden (nTr-1)*length(iLO)])';
    decoded_volatility(iCV, :, iLO) = reshape(X * B(2:end) + B(1), [nTr-1 length(iLO)]);
end

%% plot behavior of LSTM, Bayes and R-W in a sample episode
scale_factor = 5; % re-scale the axes to plot two different variables in the same space
figure, ep = 286; 
volDecoded = squeeze(nanmean(decoded_volatility(:,:,ep)));
subplot(2,1,1), 
plot(1-PR(:,ep), 'k-'), ylim([-0.1 1.1]), hold on 
plot(Pact_Bayes(:,ep), 'ro'), plot(fb(:,ep), 'kx'), xlabel('step')
plot(Vtracker(:,ep)*scale_factor, 'b-')
plot(fitAlpha_Bayes(:,2), 'r-'), title('Bayes'), legend({'true p', 'action', 'feedback', 'inferred vol', 'learning rate'})
subplot(2,1,2), 
plot(1-PR(:,ep), 'k-'), ylim([-0.1 1.1]), hold on 
plot(act_LSTM(:,ep), 'ro'), plot(fb(:,ep), 'kx'), xlabel('step')
plot(volDecoded*scale_factor, 'b-')
plot(fitAlpha_LSTM(:,2), 'r-'), title('LSTM'), legend({'true p', 'action', 'feedback', 'decoded vol', 'learning rate'})

%% plot parameter estimates for LSTM and Bayes, split by phase (stable first, volatile first, volatile second, stable second)
figure,  
barwitherr([sem(fitAlpha_LSTM(1:100,2)) sem(fitAlpha_LSTM(1:100,1)) sem(fitAlpha_LSTM(101:200,2)) sem(fitAlpha_LSTM(101:200,1))], ...
           [1 2 3 4], ...
           [mean(fitAlpha_LSTM(1:100,2)) mean(fitAlpha_LSTM(1:100,1)) nanmean(fitAlpha_LSTM(101:200,2)) nanmean(fitAlpha_LSTM(101:200,1))])
set(gca, 'YScale', 'log', 'XTick', [1 2 3 4], 'XTickLabel', {'Stable 1','Volatile 1','Volatile 2','Stable 2'})
ylim([0.1 0.7])
ylabel('learning rate')
title('LSTM')












