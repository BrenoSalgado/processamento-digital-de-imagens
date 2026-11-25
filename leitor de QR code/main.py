import cv2
import os
import webbrowser
import numpy as np
import matplotlib.pyplot as plt
from pyzbar.pyzbar import decode

def mostrar_imagens(etapas, titulos):
    plt.figure(figsize=(15, 8))
    for i in range(len(etapas)):
        plt.subplot(2, 3, i + 1)
        if len(etapas[i].shape) == 2:
            plt.imshow(etapas[i], cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(etapas[i], cv2.COLOR_BGR2RGB))
        plt.title(titulos[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def tentar_decodificar(img):
    return decode(img)

def ler_qr_code(caminho_imagem):
    if not os.path.exists(caminho_imagem):
        print("Erro: imagem nao encontrada.")
        return

    imagem = cv2.imread(caminho_imagem)

    if imagem is None:
        print("Erro: nao foi possivel carregar a imagem.")
        return

    etapas = []
    titulos = []

    etapas.append(imagem.copy())
    titulos.append("Original")

    resultados = tentar_decodificar(imagem)

    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    etapas.append(cinza)
    titulos.append("Escala de cinza")

    if not resultados:
        resultados = tentar_decodificar(cinza)

    blur = cv2.GaussianBlur(cinza, (5, 5), 0)
    etapas.append(blur)
    titulos.append("Filtro Gaussiano")

    if not resultados:
        resultados = tentar_decodificar(blur)

    _, binaria = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    etapas.append(binaria)
    titulos.append("Limiarizacao Otsu")

    if not resultados:
        resultados = tentar_decodificar(binaria)

    kernel = np.ones((10, 10), np.uint8)

    binaria_inv = cv2.bitwise_not(binaria)

    abertura_inv = cv2.morphologyEx(binaria_inv, cv2.MORPH_OPEN, kernel)
    abertura = cv2.bitwise_not(abertura_inv)

    etapas.append(abertura)
    titulos.append("Abertura")

    if not resultados:
        resultados = tentar_decodificar(abertura)

    fechamento_inv = cv2.morphologyEx(abertura_inv, cv2.MORPH_CLOSE, kernel)
    fechamento = cv2.bitwise_not(fechamento_inv)

    etapas.append(fechamento)
    titulos.append("Fechamento")

    if not resultados:
        resultados = tentar_decodificar(fechamento)

    if not resultados:
        ampliada = cv2.resize(fechamento, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        resultados = tentar_decodificar(ampliada)
    else:
        ampliada = cv2.resize(fechamento, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    if resultados:
        print("QR Code encontrado:")
        for resultado in resultados:
            texto = resultado.data.decode("utf-8")
            print(texto)

            x, y, w, h = resultado.rect
            cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 3)

            if texto.startswith("http://") or texto.startswith("https://"):
                print("Abrindo no navegador...")
                webbrowser.open(texto)

        cv2.imwrite("resultado_qr.png", imagem)
        print("Imagem com deteccao salva como resultado_qr.png")
    else:
        print("Nenhum QR Code foi encontrado na imagem.")
        cv2.imwrite("imagem_analisada.png", imagem)
        print("Imagem analisada salva como imagem_analisada.png")

    mostrar_imagens(
        [imagem, cinza, blur, binaria, abertura, fechamento],
        ["Imagem Original", "Cinza", "Gaussiano", "Otsu", "Abertura", "Fechamento"]
    )

def main():
    caminho_imagem = r"D:/Faculdade/leitura de QR code/imagens/sal-pimenta.png"
    ler_qr_code(caminho_imagem)

if __name__ == "__main__":
    main()