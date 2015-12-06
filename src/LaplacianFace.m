classdef LaplacianFace < ExtractFeature
    
    properties
        % Eigenface feature extractor
        eigenF;
        
        % Laplacian parameters
        epsilon = 10;
        t = 1;
        
        % desired dimension
        dimension = 25;
        
        % projection matrix
        W;
    end
    
    methods
        function [obj] = LaplacianFace()
            obj.eigenF = EigenFaceFeature();
            obj.eigenF.dimension = 25;
        end
        
        function init(self, images)
            A = readImages(self, images);
            A = self.eigenF.project(A);
            computeProjectionMatrix(self, A);
        end
        
        function [Ad] = extract(self, images)
            A = readImages(self, images);
            Ad = self.project(A);
        end
        
        function [Ad] = project(self, A)
            A = self.eigenF.project(A);
            Ad = A * self.W;
        end
        
        function computeProjectionMatrix(self, X)
            % Create graph with N nodes and edge between nearest neighbours
            % create node graph with weights as distance
            N = size(X,1);
            S = zeros(N);
            [idx,D] = knnsearch(X,X,'K',self.dimension);
            for i = 1 : N
                S(i,idx(i,:)) = D(i,:);
            end
            S(S > 0) = exp(-(S(S > 0) / self.t));
            
            D = diag(sum(S));
            L = D - S;
            
            % solve generalized eigen value for X'LXw = λX'DXw
            A = X' * L * X;
            B = X' * D * X;
            [V,E] = eig(A, B);
            [~,sI] = sort(diag(E));
            
            self.W = V(:, sI(1:self.dimension));
        end
        
        function constructGraph(self, X)
            S = pdist2(X,X);
            % keep only closer nodes
            S(S > self.epsilon) = 0; 
            % make distance exponential
            S(S > 0) = exp(-(S(S > 0) / self.t));
        end
    end
    
end

