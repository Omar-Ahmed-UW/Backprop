classdef BackpropLayer
    properties
        W
        b
        mostRecentN
    end

    methods
        function obj = BackpropLayer(first, second)
            % checks to generate weight vectors and bias vector or set to
            % passed in vectors.
            [fr,fc] = size(first);
            [sr,sc] = size(second);
            if fr == 1 && fc == 1 && sr == 1 && sc == 1
                obj.W = (-1 + (1+1).*rand(second, first))';
                obj.b = (-1 + (1+1).*rand(second, 1))';
            
            else
                obj.W = first;
                obj.b = second;
            end        
        end

        function [obj, a] = forwardprop(obj, P)
            % uses vectorized approach to calculate inner product of all
            % weights with inputs then adds the correct biases.
            n = (P*obj.W) + obj.b;
            a = logsig(n);
            obj.mostRecentN = n;
        end

        function [obj, output] = layersensitivity(obj, x)
            % calculates the sensitivity for this layer
            % x represents the component from the following layer (either
            % t-a or the sum of sensitivity*weight for each neuron)
            output = -2*(derivlogsig(obj.mostRecentN)*x);
        end

        
    end
end