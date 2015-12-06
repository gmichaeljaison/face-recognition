classdef EigenFaceFeature < ExtractFeature
    
    properties
        % default PCA dimension
        dimension = 7
        mean_faces
        projection_matrix
    end
    
    methods
        function init(self, images)
            A = readImages(self, images);
            computeProjectionMatrix(self, A);
        end
        
        function [Ad] = extract(self, images)
            A = readImages(self, images);
            
            Ad = self.project(A);
        end
        
        function [Ad] = project(self, A)
            N = size(A,1);
            
            A = A - repmat(self.mean_faces, N, 1);
            
            % 5. Project to lower dimension
            Ad = A * self.projection_matrix;
        end
        
        function [R] = computeProjectionMatrix(self, A)
            N = size(A,1);
            
            % 1. calculate mean
            self.mean_faces = mean(A, 1);
            
            % 2. shift to center
            A = A - repmat(self.mean_faces, N, 1);

            % 3. find Principal components in MxM space
            [evectors,~] = eig(A * A');
            evectors = evectors * A;
            
%             evectors = princomp(A);
            
            % 4. Retain top eigen vectors with maximum variance
            K = size(evectors,1);
            if (K > self.dimension)
                K = self.dimension;
            end;
            R = evectors(1:K, :)';
           
            self.projection_matrix = R;
        end
    end
    
end

