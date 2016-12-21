%by Dr. Rex Cheung
%cheung.r100@gmail.com
%This program resamples (i.e. coarsens or refines) a picture and returns the x, y, z 
%componenets and the resampled image, then plot the image for
%visualization. This is useful for multi-grid image registration to improve
%accuracy and efficiency. Note in the example with phantom, users can see
%the more pixelated figure with the coarser grid.

function resampleImage=resample3Dimage(image, factor)
%check dimension is 3D
if (ndims(image)<3)
    error('image needs to be a 3D image');
end
%put the image in an internal buffer, so that the original image is not
%destroyed during calculations
buffer=image;

%size each dimension of the image i.e. find the number of rows, number of
%columns and number of pages
[nrow ncol npage]=size(buffer);

%Use the above results to create the old grid
%IMPORTANT REMINDER NOTE: for meshgrid implementation in Matlab meshgrid(a,b,c) means
%there are a columns, b rows and c pages. Therefore put ncol in a position,
%and nrow in b position, and npage as expected in the c position

%create the original grid size
[x,y,z] = meshgrid(1:ncol,1:nrow,1:npage);

%create the new grid for resampling of factor of 2, in this case the grid
%is coarser than the original grid. It may be useful for initial
%registrations to save time, also allows pattern registration as well.
[xi,yi,zi] = meshgrid(1:factor:ncol, 1:factor:nrow, 1:factor:npage);

%interpolate the values on the new grid using matlab interp3
vi = interp3(x,y,z,buffer,xi,yi,zi); 
subplot(2,1,1);imagesc(image(:,:,round(npage/2)));title('original image');hold on;
subplot(2,1,2);imagesc(vi(:,:,round(npage/2/factor)));title('resampled image');hold on;

%return the result to the caller
resampleImage=vi;

%resampleImage=resample3Dimage(t, 2)

