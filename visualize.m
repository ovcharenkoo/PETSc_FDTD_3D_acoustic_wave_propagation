% clear all;

% addpath('/Users/ovcharoo/Software/petsc-3.7.3/arch-darwin-c-debug/lib/petsc/matlab');
% addpath('/Users/ovcharoo/Software/petsc-3.7.3/share/petsc/matlab');
% 
% fileID = fopen('binaryoutput','r');
% % A = fread(fileID);
% % A = PetscBinaryRead(fileID, 1 ,'PETSC_DOUBLE');
clear all;

tmp_Uvec;

dim = int8(round(abs(max(size(Vec_0x84000000_0))))^(1/3));
u = reshape(Vec_0x84000000_0, dim, dim, dim);

% u = resample3Dimage(u, 2);

%%
% for i=2:max(size(u))-1
%     clf; imagesc(squeeze(u(i,:,:))); title(num2str(i)); drawnow; pause(0.2);
% end

%% 3D transparent
close all;

indPatch=1:numel(u);
[F,V,C]=ind2patch(indPatch,u,'v');
kk = 0.1;
patch('Faces',F,'Vertices',V,'FaceColor','flat','CData',C,'EdgeColor','none','FaceAlpha','flat','FaceVertexAlphaData',double(C > kk * max(C)));
% patch('Faces',F,'Vertices',V,'FaceColor','flat','CData',C,'EdgeColor','none','FaceAlpha',0.5);
axis equal; view(3); axis tight; axis vis3d; grid off; 

%%

% [x y z] = ind2sub(size(u), find(u));
% % plot3(x, y, z, 'k.');
% scatter3(x, y, z,'o','filled');