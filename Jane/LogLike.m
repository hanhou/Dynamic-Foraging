function [ nLL ] = LogLike( choice_data, model_probs ) 
% returns the negative log likelihood for particular data under a particular model, with particular parameters (which are already included in model_func)
% whichTrials is a logical mask that says which trials count toward the loss

nLL = -sum(sum(log(shiftdim(choice_data==0,-1) .* (1-model_probs) + shiftdim(choice_data==1,-1) .* model_probs),3),2);  % probability of actual choice

end

