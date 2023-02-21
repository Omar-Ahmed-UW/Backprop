input0 = [-1 1 1 1 -1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 -1 1 1 1 -1];
input1 = [-1 1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1];
input2 = [1 1 1 -1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 1 1 -1 -1 -1 1 -1 -1 -1 -1 1 1 1 1];

t0 = [1 0 0];
t1 = [0 1 0];
t2 = [0 0 1];

network = BackpropNetwork(30,3, 3);

sumW1 = zeros(3,30);
sumB1 = zeros(3,1);
sumW2 = zeros(3,3);
sumB2 = zeros(3,1);

for i = 1:1000
    [network, ~] = network.networkForward(input0');
    [network, temp1, temp2] = network.networkSensitivity(t0');
    sumW1 = sumW1 + temp1*network.a0';
    sumB1 = sumB1 + temp1;
    sumW2 = sumW2 + temp2*network.a1';
    sumB2 = sumB2 + temp2;

    [network, ~] = network.networkForward(input1');
    [network, temp1, temp2] = network.networkSensitivity(t1');
    sumW1 = sumW1 + temp1*network.a0';
    sumB1 = sumB1 + temp1;
    sumW2 = sumW2 + temp2*network.a1';
    sumB2 = sumB2 + temp2;


    [network, ~] = network.networkForward(input2');
    [network, temp1, temp2] = network.networkSensitivity(t2');
    sumW1 = sumW1 + temp1*network.a0';
    sumB1 = sumB1 + temp1;
    sumW2 = sumW2 + temp2*network.a1';
    sumB2 = sumB2 + temp2;


    network = network.batchUpdateNetwork(sumW1, sumB1, sumW2, sumB2);
    sumW1 = zeros(3,30);
    sumB1 = zeros(3,1);
    sumW2 = zeros(3,3);
    sumB2 = zeros(3,1);
end

[network, test] = network.networkForward(input0');
disp(round(test))

[network, test] = network.networkForward(input1');
disp(round(test))

[network, test] = network.networkForward(input2');
disp(round(test))
