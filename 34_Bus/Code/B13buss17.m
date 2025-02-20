
% salva em uma váriavel os dados da linha C
%Ic = Iabc(:,3);
data = Ic';

% Defina a matriz original com 200.000 linhas
% Suponha que você tenha uma matriz chamada 'matriz_original' com 300.000 linhas e qualquer número de colunas
dados = data(1:200000,:);
% Número de linhas em cada parte
linhas_por_parte = 200000 / 200; % segundos

% Divida a matriz original em partes
partes = mat2cell(dados, repmat(linhas_por_parte, 1, 200), size(dados, 2));% repmat(linhas_por_parte, 1, 100)

% Crie uma nova matriz pivotando cada parte
nova_matriz = cellfun(@transpose, partes, 'UniformOutput', false);

% Converta a nova matriz de células para uma matriz normal
nova_matriz = cat(1, nova_matriz{:});


% Especificar o nome do arquivo
nome_arquivo = '19_06_20s_13Bus_200_linhas_10mile_seg.csv';


% Salvar a matriz no arquivo de texto separado por ponto e vírgula
dlmwrite(nome_arquivo, nova_matriz, ';');
