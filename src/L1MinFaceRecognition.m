classdef L1MinFaceRecognition < FaceRecognition
    %L1MINFACERECOGNITION Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        epsilon
        xTrain
        yTrain
        SCI_threshold = 0.3;
    end
    
    methods
        function [obj] = L1MinFaceRecognition(xTrain, trainLabels, epsilon)
            obj.xTrain = xTrain;
            obj.yTrain = trainLabels;
            obj.epsilon = epsilon;
        end
        
        function train()
        end
        
        function [Y] = predict(self, X)
            nX = size(X,1);
            
            A = double(self.xTrain);
            X = double(X);

            Y = cell(nX,1);
            for n = 1 : nX
                x = X(n,:)';
                x0 = A * x;

                xp = l1qc_logbarrier(x0, A', [], x, self.epsilon, 1e-3);

                nPerson = numel(unique(self.yTrain));
                residuals = zeros(nPerson,1);
                [~,~,labelInd] = unique(self.yTrain);
                delta_xps = zeros(nPerson, numel(xp));
                for i = 1 : nPerson
                    delta_xp = zeros(size(xp));

                    range = (labelInd == i);
                    delta_xp(range) = xp(range);
                    delta_xps(i,:) = delta_xp;

                    err = x - (A' * delta_xp);
                    residuals(i) = norm(err,2);
                end

                SCI = self.sparsityConcentrationIndex(delta_xps, xp);

                if (SCI > self.SCI_threshold)
                    [~,minInd] = min(residuals);
                    Y{n} = self.yTrain{minInd};
                else
                    Y{n} = 'unknown';
                end
            end
                
        end
    end
    
    methods (Static)
        function SCI = sparsityConcentrationIndex(delta_xps, xp)
            nPersons = size(delta_xps,1);
            SCIs = zeros(nPersons,1);
            for i = 1 : nPersons
                SCIs(i) = norm(delta_xps(i,:)) / norm(xp);
            end
            SCI = max(SCIs);
        end
    end
    
end
