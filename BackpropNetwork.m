classdef BackpropNetwork
    
    properties
        % Backprop layer instances
        layer1
        layer2

        % easier to keep a in two different variables for now, because it's
        % a jagged array
        %   TODO linnea: possibly use "cell array" in the future?
        a0
        a1
        a2        
        
        % storing sensitivies computed for each of the two layers
        s1
        s2

        % storing the learning rate
        learningRate
    end
    
    methods
        function obj = BackpropNetwork(input, hidden, output)
            % creates the 2 layers of the network
            % input and output are the number of input and output variables
            % hidden is the number of neurons in the hidden layer
            obj.layer1 = BackpropLayer(input, hidden);
            obj.layer2 = BackpropLayer(hidden, output);

            % todo: this was a total guess!!!
            learningRate = 0.05;
        end
        
        function [obj, temp2] = networkForward(obj,input)
            % propogates outputs forward through all layers
            % returns the final output
            obj.a0 = input;
            [obj.layer1, obj.a1] = obj.layer1.layerForward(input);
            [obj.layer2, temp2] = obj.layer2.layerForward(obj.a1);
            obj.a2 = temp2;
        end

        function [obj] = networkSensitivity(obj, targetOutput)
            % computes the sensitivties for all layers in the network using
            % the targetOutput for the output layer.
            [obj.layer2, obj.s2] = obj.layer2.layerSensitivity(-2*(targetOutput - obj.a2));
            [obj.layer1, obj.s1] = obj.layer1.layerSensitivity((obj.layer2.W'*obj.s2));
        end

        function obj = networkUpdate(obj)
            % updates the weights and sensitivities for both layers
            obj.layer1.updateLayer(obj.learningRate, obj.s1, obj.a0);
            obj.layer2.updateLayer(obj.learningRate, obj.s2, obj.a1);
        end
    end
end

