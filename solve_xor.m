%% The following is a complete example of single layer neural network-
%% from scratch using Matlab (no frameworks - no blackbox functions!)
%% @author: Hunar Ahmad Abdulrahman

%% xor data
X1=[0,0; 1,1; 1,0; 0,1];  %inputs
y=[0; 0; 1; 1];          %outputs

% model settings 
lr=0.1;
in=2; nodes=4; out =1;
% weight initialization
W1=randn(in,nodes);
W2=randn(nodes,out);
%% training loop...
for i=1:100;
    % feedforward
    z1=X1*W1;
    X2=sin(z1); % "sin" used as "activation function" for simplicity
    z2=X2*W2;
    yhat=sin(z2);
    % backpropagation
    e1 = y - yhat
    delta2 = e1.*cos(z2); % "cos" is derivative of "sin"
    e2 = (delta2*W2')
    delta1 = e2.* cos(z1);
    dW2 = X2'* delta2;
    dW1 = X1'* delta1;
    W2=W2+ dW2 *lr;
    W1=W1+ dW1 *lr;
    % optional line ( squared error for visualization);
    mse(i)=mean((y-yhat).^2);
end
%% output & error visualization
plot(mse); title('MSE');
[yhat, y] % compare the predictions with the true labels
