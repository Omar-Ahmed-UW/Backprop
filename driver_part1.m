input0 = [-1 1 1 1 -1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 -1 1 1 1 -1];
input1 = [-1 1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1];
input2 = [1 1 1 -1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 1 1 -1 -1 -1 1 -1 -1 -1 -1 1 1 1 1];

t0 = [1 0 0];
t1 = [0 1 0];
t2 = [0 0 1];

network = OneLayerBackpropNetwork(30, 3);

for i = 1:1000
    [network, a0] = network.networkForwardOneLayer(input0');
    network = network.networkSensitivityOneLayer(t0');
    network = network.networkUpdateOneLayer();

    [network, a1] = network.networkForwardOneLayer(input1');
    network = network.networkSensitivityOneLayer(t1');
    network = network.networkUpdateOneLayer();

    [network, a2] = network.networkForwardOneLayer(input2');
    network = network.networkSensitivityOneLayer(t2');
    network = network.networkUpdateOneLayer();
end

[network, test] = network.networkForwardOneLayer(input0');
disp(round(test))

[network, test] = network.networkForwardOneLayer(input1');
disp(round(test))

[network, test] = network.networkForwardOneLayer(input2');
disp(round(test))
