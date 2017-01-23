% clear all;

% addpath('/Users/ovcharoo/Software/petsc-3.7.3/arch-darwin-c-debug/lib/petsc/matlab');
% addpath('/Users/ovcharoo/Software/petsc-3.7.3/share/petsc/matlab');
% 
% fileID = fopen('binaryoutput','r');
% % A = fread(fileID);
% % A = PetscBinaryRead(fileID, 1 ,'PETSC_DOUBLE');
clear all;
close all;

addpath('./subroutines');

% h1 = figure('units','normalized','outerposition',[0 0 1 1]);
h1 = figure;
WinOnTop(h1);

for ii = 1:35
    kf = ii * 2;
    name = ['../wavefields/10Hz_128/tmp_Bvec_' num2str(kf)];
    run(name);

    %%
    dim = int8(round(abs(max(size(Vec_0x84000004_0))))^(1/3));
    u = reshape(Vec_0x84000004_0, dim, dim, dim);

    % u = resample3Dimage(u, 2);

    %%
%     close all;
%     
%     for i=2:max(size(u))-1
%         clf; imagesc(squeeze(u(i,:,:))); title(num2str(i)); colorbar; drawnow; pause(0.2);
%     end

%%
%     close all;
%     
%     for i=round(max(size(u))/2):round(max(size(u))/2)
%         clf; imagesc(squeeze(u(i,:,:))); title(num2str(i)); colorbar; drawnow; pause(0.2);
%     end


    %% 3D transparent
%     close all;
% u1 = u;
% u(1:round(end/2),:,:)=0;
    indPatch=1:numel(u);
    [F,V,C]=ind2patch(indPatch,u,'v');
    kk = 0.01;
    clf; patch('Faces',F,'Vertices',V,'FaceColor','flat','CData',C,'EdgeColor','none','FaceAlpha','flat','FaceVertexAlphaData',double(C > kk * max(C)));
    % patch('Faces',F,'Vertices',V,'FaceColor','flat','CData',C,'EdgeColor','none','FaceAlpha',0.5);
    axis equal; view(3); axis tight; axis vis3d; grid off; 
    title(['Step ' num2str(kf) ' of 121']); 
    L = get(gca,'Xlim');
    set(gca, 'Xtick',linspace(L(1),L(2),5));
    set(gca,'xticklabel',[0 2 4 6 8]);
    L = get(gca,'ylim');
    set(gca, 'ytick',linspace(L(1),L(2),5));
    set(gca,'yticklabel',[0 2 4 6 8]);
    L = get(gca,'zlim');
    set(gca, 'ztick',linspace(L(1),L(2),5));
    set(gca,'zticklabel',[0 2 4 6 8]);
    xlabel('km');
    ylabel('km');
    zlabel('km');
    c=colorbar;
    ylabel(c,'Amplitude')
    set(gca,'FontSize',14)
    drawnow;
%%
    saveas(gcf,['step' num2str(kf) '.png']);
    %%

    % [x y z] = ind2sub(size(u), find(u));
    % % plot3(x, y, z, 'k.');
    % scatter3(x, y, z,'o','filled');
%     input('?');
% pause(1)
end
