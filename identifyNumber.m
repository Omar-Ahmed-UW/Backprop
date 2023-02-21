function output = identifyNumber(testNumber)
%IDENTIFYNUMBER Converts a 3-entry long vector back into a number from 0-2
%   Returns -1 if the vector wasn't an exact match for any of our inputs

p0 = [1; 0; 0];
p1 = [0; 1; 0];
p2 = [0; 0; 1];

if testNumber == p0
    output = 0;
elseif testNumber == p1
    output = 1;
elseif testNumber == p2
    output = 2;
else
    output = -1;
end

end

