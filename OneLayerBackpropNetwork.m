classdef OneLayerBackpropNetwork
    
    properties
        % Backprop layer instances
        layer1

        % easier to keep a in two different variables for now, because it's
        % a jagged array
        %   TODO linnea: possibly use "cell array" in the future?
        a0
        a1       
        
        % storing sensitivies computed for each of the two layers
        s1

        % storing the learning rate
        learningRate
    end
    
    methods
        function obj = OneLayerBackpropNetwork(input, output)
            % creates the 2 layers of the network
            % input and output are the number of input and output variables
            % hidden is the number of neurons in the hidden layer
            obj.layer1 = BackpropLayer(input, output);

            % todo: this was a total guess!!!
            obj.learningRate = 0.05;
        end
        
        function [obj, temp] = networkForwardOneLayer(obj,input)
            % propogates outputs forward through all layers
            % returns the final output
            obj.a0 = input;
            [obj.layer1, temp] = obj.layer1.layerForward(input);
            obj.a1 = temp;
        end

        function [obj] = networkSensitivityOneLayer(obj, targetOutput)
            % computes the sensitivties for all layers in the network using
            % the targetOutput for the output layer.
            [obj.layer1, obj.s1] = obj.layer1.layerSensitivity(-2*(targetOutput - obj.a1));
        end

        function obj = networkUpdateOneLayer(obj)
            % updates the weights and sensitivities for both layers
            obj.layer1 = obj.layer1.updateLayer(obj.learningRate, obj.s1, obj.a0);
        end
    end
end

