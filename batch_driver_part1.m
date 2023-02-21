input0 = [-1 1 1 1 -1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 -1 1 1 1 -1]';
input1 = [-1 1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1]';
input2 = [1 1 1 -1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 1 1 -1 -1 -1 1 -1 -1 -1 -1 1 1 1 1]';

inputs = [input0 input1 input2];

t0 = [1; 0; 0];
t1 = [0; 1; 0];
t2 = [0; 0; 1];

ts = [t0 t1 t2];



sumW1 = zeros(3,30);
sumB1 = zeros(3,1);
sumW2 = zeros(3,3);
sumB2 = zeros(3,1);

count = zeros(3, 3);

for j = 1:3
    network = BackpropNetwork(30,3, 3);
    for i = 1:1000
        for idx = 1:j
            [network, returned] = network.networkForward(inputs(:, idx));
            [network, temp1, temp2] = network.networkSensitivity(ts(:, idx));
            sumW1 = sumW1 + temp1*network.a0';
            sumB1 = sumB1 + temp1;
            sumW2 = sumW2 + temp2*network.a1';
            sumB2 = sumB2 + temp2;
        end
    
        network = network.batchUpdateNetwork(sumW1, sumB1, sumW2, sumB2);
        sumW1 = zeros(3,30);
        sumB1 = zeros(3,1);
        sumW2 = zeros(3,3);
        sumB2 = zeros(3,1);
    end

     for k = 1:3
         pass = 0;
         fail = 0;
         for m = 1:j %For each number we have saved
             for n = 1:100 %100 trials
                 newVec = addNoise(inputs(:, m), (k-1)*4);
                 [~, newOut] = network.networkForward(newVec);
                 if identifyNumber(round(newOut)) == m-1
                    pass = pass + 1;
                 else
                    fail = fail + 1;
                 end
             end
             count(k, j) = pass/(fail+pass);
         end
     end

end

x = [0 1 2 3];
blank = [1 1 1];
count = [blank; count];


figure
hold on

plot(x, count(:, 1), "r");
plot(x, count(:, 2), "g");
plot(x, count(:, 3), "b");
xticks([1 2 3]);
xlim([0 3]);
ylim([0.5 1.1]);
xlabel('Number of digits stored');
ylabel('Classification accuracy');
legend('0 pixels of noise added', '4 pixels of noise added', '8 pixels of noise added');
hold off
