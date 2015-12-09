classdef L1MinFaceRecognition < FaceRecognition
    %L1MINFACERECOGNITION Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        epsilon
        xTrain
        yTrain
        SCI_threshold = 0.4;
    end
    
    methods
        function [obj] = L1MinFaceRecognition(xTrain, yTrain, epsilon)
            obj.xTrain = normr(xTrain);
            obj.yTrain = yTrain;
            obj.epsilon = epsilon;
            
%             s = sum(xTrain, 2);
%             obj.xTrain = xTrain ./ repmat(s, 1, size(xTrain,2));
        end
        
        function train()
        end
        
        function [Y] = predict(self, X)
            nX = size(X,1);
            X = normr(X);
%             s = sum(abs(X), 2);
%             X = X ./ repmat(s, 1, size(X,2));
            
            A = double(self.xTrain);
            X = double(X);

            Y = cell(nX,1);
            for n = 1 : nX
                x = X(n,:)';
                x0 = A * x;

%                 xp = l1qc_logbarrier(x0, A', [], x, self.epsilon, 1e-3);
                xp = l1qc_logbarrier(x0, A', [], x, 0.005, 1e-1);
%                 xp = l1eq_pd(x0, A', [], x, 1e-3);

                nPerson = numel(unique(self.yTrain));
                residuals = zeros(nPerson,1);
                [labels,~,labelInd] = unique(self.yTrain);
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
                disp('residuals: '); disp(residuals);
                disp('SCI: '); disp(SCI);

                if (SCI > self.SCI_threshold)
                    [~,minInd] = min(residuals);
                    Y{n} = labels{minInd};
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
                SCIs(i) = norm(delta_xps(i,:),1) / norm(xp,1);
            end
            SCI = max(SCIs);
            
            SCI = (nPersons/(nPersons-1)) * SCI - 1/(nPersons-1);
        end
    end
    
end
