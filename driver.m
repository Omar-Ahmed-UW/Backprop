x = [20, 30, 40, 32];
y = zeros(size(x));

for j = 1:4 % testing 4 different values
    scores = zeros(1, 3);
    for i = 1:3 % 3 trials each
        scores(i) = testPart2(10, 100, x(j));
    end
    sort(scores);
    y(j) = scores(2); % using the median of 3 trials
end

figure
hold on
plot(x, y);
xlabel('Batch size');
ylabel('Classification accuracy');
hold off