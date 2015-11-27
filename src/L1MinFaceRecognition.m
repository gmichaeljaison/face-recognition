classdef L1MinFaceRecognition < FaceRecognition
    %L1MINFACERECOGNITION Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        epsilon
        trainingFeatures
        trainingLabels
    end
    
    methods
        function [obj] = L1MinFaceRecognition(xTrain, trainLabels, epsilon)
            obj.trainingFeatures = xTrain;
            obj.trainingLabels = trainLabels;
            obj.epsilon = epsilon;
        end
        
        function train()
        end
        
        function [label] = predict(self, y)
            A = double(self.trainingFeatures);
            y = double(y);

            x0 = A * y;
            
            xp = l1qc_logbarrier(x0, A', [], y, self.epsilon, 1e-3);
            
            nPerson = numel(unique(self.trainingLabels));
            residuals = zeros(nPerson,1);
            [~,~,labelInd] = unique(self.trainingLabels);
            delta_xps = zeros(nPerson, numel(xp));
            for i = 1 : nPerson
                delta_xp = zeros(size(xp));
                
                range = (labelInd == i);
                delta_xp(range) = xp(range);
                delta_xps(i,:) = delta_xp;

                err = y - (A' * delta_xp);
                residuals(i) = norm(err,2);
            end
            
            SCI = self.sparsityConcentrationIndex(delta_xps, xp);
            
            if (SCI > 0.4)
                [~,minInd] = min(residuals);
                label = self.trainingLabels(minInd);
            else
                label = 'unknown';
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
