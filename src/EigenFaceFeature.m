classdef EigenFaceFeature < ExtractFeature
    
    properties
        % default PCA dimension
        dimension = 144
        mean_faces
        projection_matrix
    end
    
    methods
        function [Ad] = extract(self, images)
            A = readImages(self, images);
            
            Ad = self.project(A);
        end
        
        function [Ad] = project(self, A)
            if ~isvector(A)
                computeProjectionMatrix(self, A);
            else
                A = A - self.mean_faces;
            end
            
            % 5. Project to lower dimension
            Ad = A * self.projection_matrix;
        end
        
        function [R] = computeProjectionMatrix(self, A)
            N = size(A,1);
            
            % 1. calculate mean
            self.mean_faces = mean(A, 1);
            
            % 2. shift to center
            A = A - repmat(self.mean_faces, N, 1);

            % 3. find Principal components
            evectors = princomp(A);
            
            % 4. Retain top eigen vectors with maximum variance
            R = evectors(:, 1:self.dimension);
           
            self.projection_matrix = R;
        end
    end
    
end

