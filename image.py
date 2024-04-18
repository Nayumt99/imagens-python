import cv2
import os
import logging

# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def redimensionar_imagem(imagem, largura=None, altura=None):
    """
    Redimensiona a imagem para a largura e/ou altura desejadas.

    Args:
        imagem: Imagem a ser redimensionada.
        largura (int): Largura desejada da imagem redimensionada.
        altura (int): Altura desejada da imagem redimensionada.

    Returns:
        Imagem redimensionada.
    """
    try:
        if largura is None and altura is None:
            return imagem
        if largura is None:
            proporcao = altura / float(imagem.shape[0])
            dimensao = (int(imagem.shape[1] * proporcao), altura)
        else:
            proporcao = largura / float(imagem.shape[1])
            dimensao = (largura, int(imagem.shape[0] * proporcao))
        return cv2.resize(imagem, dimensao, interpolation=cv2.INTER_AREA)
    except Exception as e:
        logger.error(f"Erro ao redimensionar a imagem: {e}")
        return None

def aplicar_filtro_grayscale(imagem):
    """
    Aplica o filtro de escala de cinza na imagem.

    Args:
        imagem: Imagem a ser convertida para escala de cinza.

    Returns:
        Imagem convertida para escala de cinza.
    """
    try:
        return cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        logger.error(f"Erro ao aplicar o filtro de escala de cinza: {e}")
        return None

def carregar_imagem(caminho):
    """
    Carrega uma imagem do caminho fornecido.

    Args:
        caminho (str): Caminho do arquivo de imagem.

    Returns:
        Imagem carregada.
    """
    try:
        imagem = cv2.imread(caminho)
        if imagem is None:
            logger.warning("A imagem não pôde ser carregada. Verifique o formato ou o caminho do arquivo.")
        return imagem
    except Exception as e:
        logger.error(f"Erro ao carregar a imagem: {e}")
        return None

def main():
    try:
        # Solicitar o caminho da imagem ao usuário
        caminho_imagem = input("Digite o caminho da imagem: ")
        if not os.path.isfile(caminho_imagem):
            logger.error("O arquivo de imagem especificado não existe.")
            return

        # Carregar a imagem
        imagem = carregar_imagem(caminho_imagem)
        if imagem is None:
            return

        # Redimensionar a imagem para 300 de largura
        imagem_redimensionada = redimensionar_imagem(imagem, largura=300)

        # Aplicar filtro de escala de cinza na imagem
        imagem_em_grayscale = aplicar_filtro_grayscale(imagem)

        # Mostrar as imagens resultantes
        if imagem_redimensionada is not None:
            cv2.imshow('Imagem Redimensionada', imagem_redimensionada)
        if imagem_em_grayscale is not None:
            cv2.imshow('Imagem em Grayscale', imagem_em_grayscale)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        logger.error(f"Erro ao processar a imagem: {e}")

if __name__ == "__main__":
    main()
