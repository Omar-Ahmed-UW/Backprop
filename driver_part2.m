load('mnist.mat');
test
training
network = BackpropNetwork(784, 600, 10);


for i = 1:60000
    [network, ~] = network.networkForward(training.images(:, i));
    network = network.networkSensitivity(training.labels(:, i));
    network = network.networkUpdate();
end


[network, temp] = network.networkForward(test.images(:, 1));
temp

test.labels(:, 1)