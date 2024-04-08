function [ Pact ] = ModelRW(fb, mags, alpha, beta )
% this is hard-coded for the one-arm volatility task
% we'll allow it to know about the perfect anti-correlation.
% alpha is the learning rate. assume vectorized in rows.
% beta is inverse temp. assume vectorized in rows.
% gamma is risk parameter. assume vectorized in rows.

nTr = size(fb,1);   % number of trials per episode
nEp = size(fb,2);  % number of episodes

nP = size(alpha,1);

Pact = nan(nP, nTr, nEp);

for iEp=1:nEp
    V = 0.5*ones(nP,1);

    for iTr=1:nTr
        %% make choice
        G0 = (1-V) .* mags(iTr, iEp, 1);
        G1 = V .* mags(iTr, iEp, 2);
        Pact(:,iTr,iEp) = 1 ./ (1 + exp(-beta.*(G1-G0)));
        
        %% update V
        delta = fb(iTr,iEp) - V;   % feedback that (1/2)-is-correct
        V = V + alpha .* delta;
    end
end




end

