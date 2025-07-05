%% --- Parámetros geométricos ------------------------------------------
R  = 1.0;      % radio del cilindro
h  = 4.0;      % altura del cilindro (de z = 0 a z = h)
k  = 0.5;      % pendiente dz/dθ  (determina el paso de la hélice)
nC = 80;       % resolución del cilindro (a más alto, malla más densa)
nH = 400;      % número de puntos en la hélice

%% --- Malla del cilindro ----------------------------------------------
thetaC = linspace(0, 2*pi, nC);
zC     = linspace(0, h, nC);
[Theta,Z] = meshgrid(thetaC, zC);

Xc = R * cos(Theta);
Yc = R * sin(Theta);
Zc = Z;

%% --- Curva hélice -----------------------------------------------------
thetaH = linspace(0, 2*pi, nH);      % una vuelta completa
Xh = R * cos(thetaH);
Yh = R * sin(thetaH);
Zh = k * thetaH;                     % z = k θ
% Si quieres que la hélice cubra toda la altura, ajusta thetaH:
% thetaH = linspace(0, h/k, nH);

%% --- Gráfica ----------------------------------------------------------
figure
surf(Xc, Yc, Zc, 'FaceAlpha',0.3, 'EdgeColor','none'); % cilindro semitransparente
colormap([0.8 0.8 1])           % color suave
hold on
plot3(Xh, Yh, Zh, 'LineWidth',2.5, 'Color',[0.85 0.1 0.1]); % hélice
hold off
axis equal
grid on
xlabel('x'), ylabel('y'), zlabel('z')
title('Geodésica (hélice) sobre un cilindro')
view(45,25)                      % ángulo de cámara agradable
