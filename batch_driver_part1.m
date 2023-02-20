input0 = [-1 1 1 1 -1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 -1 1 1 1 -1];
input1 = [-1 1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1];
input2 = [1 1 1 -1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 1 1 -1 -1 -1 1 -1 -1 -1 -1 1 1 1 1];

t0 = [1 0 0];
t1 = [0 1 0];
t2 = [0 0 1];

network = OneLayerBackpropNetwork(30, 3);

sum = zeros(3,30);
sumS = zeros(3,1);

for i = 1:1000
    [network, ~] = network.networkForwardOneLayer(input0');
    [network, temp] = network.networkSensitivityOneLayer(t0');
    sum = sum + temp*input0;
    sumS = sumS + temp;

    [network, ~] = network.networkForwardOneLayer(input1');
    [network, temp] = network.networkSensitivityOneLayer(t1');
    sum = sum + temp*input1;
    sumS = sumS + temp;

    [network, ~] = network.networkForwardOneLayer(input2');
    [network, temp] = network.networkSensitivityOneLayer(t2');
    sum = sum + temp*input2;
    sumS = sumS + temp;

    network = network.batchUpdate(sum, sumS);
    sum = zeros(3,30);
    sumS = zeros(3,1);
end

[network, test] = network.networkForwardOneLayer(input0');
disp(test)

[network, test] = network.networkForwardOneLayer(input1');
disp(test)

[network, test] = network.networkForwardOneLayer(input2');
disp(test)
