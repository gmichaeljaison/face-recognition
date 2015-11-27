classdef EigenFaceFeature < ExtractFeature
    %EIGENFACEFEATURE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        img_size = [60 60];
        dimension = 144
        mean_faces
        projection_matrix
    end
    
    methods
        function [Ad] = extract(self, images)
            if iscell(images)
                A = readImages(self, images);
            else
                A = processInputImage(self, images);
            end
%             A = processInputImage(self, images);
            
            if ~isvector(A)
                computeProjectionMatrix(self, A);
            else
                A = A - self.mean_faces;
            end
            
            % 5. Project to lower dimension
            Ad = A * self.projection_matrix;
        end
        
        function A = readImages(self, images)
            N = numel(images);
            A = zeros(N, prod(self.img_size));
            for i = 1 : N
                img = images{i};
                A(i,:) = processInputImage(self, img);
            end
        end
        
        function I = processInputImage(self, img)
            if (size(img,3) > 1)
                img = rgb2gray(img);
            end
            img = imresize(img, self.img_size);
            img = im2double(img);
            I = img(:)';
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

