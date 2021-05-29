% tv_img_interp.m
% Total variation image interpolation.
% EE364a
% Defines m, n, Uorig, Known.

% Load original image.
Uorig = double(imread('tv_img_interp.png'));

[m, n] = size(Uorig);

% Create 50% mask of known pixels.
rand('state', 1029);
Known = rand(m,n) > 0.5;

% Placeholder:
Ul2 = ones(m, n);
Utv = ones(m, n);

cvx_begin
    variable Ul2(m, n);
    Ul2(Known) == Uorig(Known); 
    U_x = Ul2(1:end,2:end) - Ul2(1:end,1:end-1);
    U_y = Ul2(2:end,1:end) - Ul2(1:end-1,1:end);
    minimize(norm([U_x(:); U_y(:)], 2));
cvx_end

cvx_begin
    variable Utv(m, n);
    Utv(Known) == Uorig(Known); % Fix known pixel values.
    Uـx = Utv(1:end,2:end) - Utv(1:end,1:end-1); % x (horiz) differences
    Uـy = Utv(2:end,1:end) - Utv(1:end-1,1:end); % y (vert) differences
    minimize(norm([Uـx(:); Uـy(:)], 1)); % tv roughness measure
cvx_end

% Graph everything.
figure(1); cla;
colormap gray;

subplot(221);
imagesc(Uorig)
title('Original image');
axis image;

subplot(222);
imagesc(Known.*Uorig + 256-150*Known);
title('Obscured image');
axis image;

subplot(223);
imagesc(Ul2);
title('l_2 reconstructed image');
axis image;

subplot(224);
imagesc(Utv);
title('Total variation reconstructed image');
axis image;
