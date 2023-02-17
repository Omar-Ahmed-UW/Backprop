classdef BackpropNetwork
    
    properties
        layer1
        layer2

        % easier to keep a in two different variables for now, because it's
        % a jagged array
        %   possibly use "cell array" in the future?
        a1
        a2

        s
    end
    
    methods
        function obj = BackpropNetwork(input, hidden, output)
            % creates the 2 layers of the network
            % input and output are the number of input and output variables
            % hidden is the number of neurons in the hidden layer
            obj.layer1 = BackpropLayer(input, hidden);
            obj.layer2 = BackpropLayer(hidden, output);
        end
        
        function [obj, temp2] = forward(obj,input)
            % propogates outputs forward through all layers
            % returns the final output
            [obj.layer1, obj.a1] = obj.layer1.forwardprop(input);
            [obj.layer2, temp2] = obj.layer2.forwardprop(obj.a1);
            obj.a2 = temp2;
        end
    end
end

