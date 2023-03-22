function [A ,B ,c , Fit ]= hmm_rosen (m ,n ,O , MaxIter )

% Parameters i n i ti a l iz at io n ...
Fit = zeros (1 , MaxIter );
% I n it ia l i za ti on ( random ) of the matrices A , B and c
A= rand (m ,m); A=A ./( A* ones (m ,1) * ones (1 , m) );
B= rand (m ,n); B=B ./( B* ones (n ,1) * ones (1 , n) );
c= rand (1 , m); c=c ./( c* ones (m ,1) ) ;

% creates x_A , x_N and x_c
x_A = reshape (A'  ,m*m ,1) ;
x_B = reshape (B' ,m*n ,1) ;
x_c =c.';
% generates the constraint matrices : QA , QB and vector QI
QA = kron ( eye (m) , ones (1 , m)) ;
QB = kron ( eye (m) , ones (1 , n)) ;
QI = ones (1 , m );

% generates the constraint matrices : bA , bB and vector bc
bA = ones (m*m ,1) ;
bB = ones (m*n ,1) ;
bc =1;

% generates the constraint matrices : RA , RB and vector Rc
RA =- eye (m*m );
RB =- eye (m*n );
Rc =- eye (m);

% generates the constraint matrices : cA , cB and vector cc
cA = zeros (m*m ,1) ;
cB = zeros (m*n ,1) ;
cc = zeros (m ,1) ;

% I n it ia l i za ti o n Phase : A
[ A_Aa , A_Ai , A_bi , A_Nk ]= MatGradRest (RA ,cA ,QA , x_A );

% I n it ia l i za ti o n Phase : B
[ B_Aa , B_Ai , B_bi , B_Nk ]= MatGradRest (RB ,cB ,QB , x_B );

% I n it ia l i za ti o n Phase : c
[ c_Aa , c_Ai , c_bi , c_Nk ]= MatGradRest (Rc ,cc ,QI , x_c );

% > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >
for j =1: MaxIter ,
    % L - gradient calculation
    [~ , dLdA ]= gradientLA_norm (A ,B ,O ,c) ;
    [~ , dLdB ]= gradientLB_norm (A ,B ,O ,c) ;
    [~,  dLdc ]= gradientLC (A ,B ,O ,c) ;
    % transform matrix into vector ...( careful with the signal ...)
    row_dLdA = - reshape ( dLdA' ,m*m ,1) ;
    row_dLdB = - reshape ( dLdB' ,m*n ,1) ;
    row_dLdc = - dLdc.';
    % HMM parameters
    Theta {1}= A; Theta {2}= B ; Theta {3}= c;
    % Rosen algorithm ...
    [ A_Aa2 , A_Ai2 , A_bi2 , A_Nk2 , x_A2 ]=...
        one_shot_rosen ( 'gradientLA_norm' , Theta ,O ,RA , cA ,QA ,bA , A_Aa , A_Ai , A_bi , A_Nk , x_A , row_dLdA ,1) ;
    [ B_Aa2 , B_Ai2 , B_bi2 , B_Nk2 , x_B2 ]=...
        one_shot_rosen ( 'gradientLB_norm' , Theta ,O ,RB , cB ,QB ,bB , B_Aa , B_Ai ,B_bi , B_Nk , x_B , row_dLdB ,2) ;
    [ c_Aa2 , c_Ai2 , c_bi2 , c_Nk2 , x_c2 ]=...
        one_shot_rosen ( 'gradientLC' , Theta ,O ,Rc ,cc ,QI ,bc , c_Aa , c_Ai , c_bi , c_Nk , x_c , row_dLdc ,3) ;
    % Update Matrices ...
    A_Aa = A_Aa2 ; A_Ai = A_Ai2 ; A_bi = A_bi2 ; A_Nk = A_Nk2 ; x_A = x_A2 ;
    B_Aa = B_Aa2 ; B_Ai = B_Ai2 ; B_bi = B_bi2 ; B_Nk = B_Nk2 ; x_B = x_B2 ;
    c_Aa = c_Aa2 ; c_Ai = c_Ai2 ; c_bi = c_bi2 ; c_Nk = c_Nk2 ; x_c = x_c2 ;
    % Rebuild A , B and c ...
    A= reshape ( x_A ,m ,m ).';
    B= reshape ( x_B ,n ,m ).';
    c= x_c.';
    % Determines the performance of new parameters ...
    [~ , P ]= forward_algorithm_norm (A ,B ,O ,c);
    disp ([ ' Iteration -- ' num2str(j) ' FIT -- ' num2str(P) ]) ;
    Fit (j )=P;
    % < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < Repetition ...
end
end