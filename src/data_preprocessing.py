import os
import random
import shutil
import cv2

def create_train_valid_test_folders(base_folder: str) -> None:
    """
    Cria as subpastas 'train', 'valid', e 'test' dentro da pasta base fornecida, separadas por categoria (Autistic, Non_Autistic).

    Args:
        base_folder (str): Caminho da pasta base onde as subpastas serão criadas.

    Returns:
        None: Esta função não retorna valor. Ela apenas cria as pastas.

    Exemplo:
        create_train_valid_test_folders("../data/data_processing/data_M-C-A")
        # Cria as pastas ../data/data_processing/data_M-C-A/train/Autistic, ../data/data_processing/data_M-C-A/train/Non_Autistic, e assim por diante.
    """
    categories = ['Autistic', 'Non_Autistic']
    for subfolder in ['train', 'valid', 'test']:
        for category in categories:
            folder_path = os.path.join(base_folder, subfolder, category)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
    print(f"Pasta base criada com subpastas 'train', 'valid', 'test' e categorias 'Autistic', 'Non_Autistic' em: {base_folder}")


def split_data(source_folder: str, train_folder: str, valid_folder: str, test_folder: str, valid_ratio: float = 0.2, train_ratio: float = 0.8) -> None:
    """
    Divide as imagens da pasta de origem em dois grupos:
    1) 80% para treino e teste, 20% para validação.
    2) Dos 80% para treino e teste, divide-os em 80% para treino e 20% para teste.

    Args:
        source_folder (str): Caminho da pasta contendo as imagens a serem divididas.
        train_folder (str): Caminho da pasta onde as imagens de treino serão armazenadas.
        valid_folder (str): Caminho da pasta onde as imagens de validação serão armazenadas.
        test_folder (str): Caminho da pasta onde as imagens de teste serão armazenadas.
        valid_ratio (float): Proporção de imagens a serem usadas para validação (padrão é 0.2).
        train_ratio (float): Proporção de imagens a serem usadas para treino (padrão é 0.8).

    Returns:
        None: Esta função não retorna valor. Ela apenas move os arquivos para as pastas correspondentes.

    Exemplo:
        split_data("../data/raw_M-C-A/Autistic", "../data/data_processing/data_M-C-A/train/Autistic", "../data/data_processing/data_M-C-A/valid/Autistic", "../data/data_processing/data_M-C-A/test/Autistic")
        # Move as imagens para as pastas apropriadas conforme as proporções fornecidas.
    """
    categories = ['Autistic', 'Non_Autistic']
    
    for category in categories:
        category_folder = os.path.join(source_folder, category)
        
        # Verifica se a pasta de categoria existe
        if not os.path.exists(category_folder):
            print(f"A pasta de categoria '{category}' não existe em {category_folder}. Verifique o caminho.")
            continue
        
        files = [f for f in os.listdir(category_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(files)  # Embaralha os arquivos para garantir a aleatoriedade

        total_files = len(files)
        
        # Divide 80% para treino e teste, 20% para validação
        valid_size = int(valid_ratio * total_files)
        valid_files = files[:valid_size]
        remaining_files = files[valid_size:]
        
        # Divide 80% restante em treino (80%) e teste (20%)
        train_size = int(0.8 * len(remaining_files))
        test_size = len(remaining_files) - train_size
        train_files = remaining_files[:train_size]
        test_files = remaining_files[train_size:]

        # Função para copiar os arquivos para as pastas de destino
        def copy_files(files: list, destination_folder: str) -> None:
            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)
            for file in files:
                src_path = os.path.join(category_folder, file)
                dst_path = os.path.join(destination_folder, file)
                try:
                    shutil.copy2(src_path, dst_path)
                    print(f"Imagem copiada: {file} de {src_path} para {dst_path}")
                except Exception as e:
                    print(f"Erro ao copiar {file}: {e}")

        # Copiar os arquivos para as pastas de treino, validação e teste
        copy_files(train_files, os.path.join(train_folder, category))
        copy_files(valid_files, os.path.join(valid_folder, category))
        copy_files(test_files, os.path.join(test_folder, category))

    print(f"Dados divididos para as categorias 'Autistic' e 'Non_Autistic' em treino, validação e teste.")


def apply_data_augmentation(train_folder: str, augmented_folder: str) -> None:
    """
    Aplica o *data augmentation* nas imagens da pasta de treino, criando versões espelhadas das imagens.

    Args:
        train_folder (str): Caminho da pasta onde as imagens de treino estão armazenadas.
        augmented_folder (str): Caminho da pasta onde as imagens aumentadas (espelhadas) serão armazenadas.

    Returns:
        None: Esta função não retorna valor. Ela apenas cria as imagens aumentadas e as salva.

    Exemplo:
        apply_data_augmentation("../data/data_processing/data_M-C-A/train", "../data/data_processing/data_M-C-A-E/train")
        # Cria versões espelhadas das imagens em ../data/data_processing/data_M-C-A-E/train.
    """
    categories = ['Autistic', 'Non_Autistic']
    
    if not os.path.exists(augmented_folder):
        os.makedirs(augmented_folder)

    # Para cada categoria, aplica o aumento de dados (espelhamento)
    for category in categories:
        category_folder = os.path.join(train_folder, category)
        augmented_category_folder = os.path.join(augmented_folder, category)

        if not os.path.exists(augmented_category_folder):
            os.makedirs(augmented_category_folder)
        
        train_files = [f for f in os.listdir(category_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for file in train_files:
            image_path = os.path.join(category_folder, file)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Não foi possível ler a imagem {file}. Ignorando...")
                continue
            
            # Aplica o espelhamento (flip horizontal)
            flipped_image = cv2.flip(image, 1)  # 1 para flip horizontal

            # Cria o novo nome para a imagem espelhada
            flipped_filename = f"{os.path.splitext(file)[0]}E{os.path.splitext(file)[1]}"
            augmented_image_path = os.path.join(augmented_category_folder, flipped_filename)

            try:
                cv2.imwrite(augmented_image_path, flipped_image)
                print(f"Imagem espelhada salva: {augmented_image_path}")
            except Exception as e:
                print(f"Erro ao salvar a imagem espelhada {file}: {e}")


def process_data_folders() -> None:
    """
    Função principal que orquestra o processo de criação das pastas, divisão dos dados e aplicação de *data augmentation*.

    Args:
        None: Esta função não recebe argumentos.

    Returns:
        None: Esta função não retorna valor. Ela executa o processo completo de organização dos dados e aumento de dados.

    Exemplo:
        process_data_folders()
        # Executa todo o processo de separação dos dados e aumento (espelhamento) das imagens.
    """
    # Caminhos das pastas de entrada e saída
    raw_data_folder = "../data/data_exclusao_manual"
    processed_data_folder = "../data/data_processing/data_exclusao_manual"
    augmented_data_folder = "../data/data_processing/data_exclusao_manual_E"
    
    # Criando as pastas de treino, validação e teste para as categorias
    create_train_valid_test_folders(processed_data_folder)
    create_train_valid_test_folders(augmented_data_folder)
    
    # Caminhos para as subpastas de treino, validação e teste
    train_folder = os.path.join(processed_data_folder, "train")
    valid_folder = os.path.join(processed_data_folder, "valid")
    test_folder = os.path.join(processed_data_folder, "test")
    
    augmented_train_folder = os.path.join(augmented_data_folder, "train")
    
    # Passo 1: Separar os dados em treino, validação e teste
    split_data(raw_data_folder, train_folder, valid_folder, test_folder)
    
    # Passo 2: Aplicar *data augmentation* (espelhamento) nas imagens de treino e salvar com sufixo 'E'
    apply_data_augmentation(train_folder, augmented_train_folder)
    
    # Passo 3: Copiar as imagens de valid e test para a pasta correspondente em data_M-C-A-E
    copy_files(valid_folder, os.path.join(augmented_data_folder, "valid"))
    copy_files(test_folder, os.path.join(augmented_data_folder, "test"))
    
    print("Processamento concluído!")


def copy_files(source_folder: str, destination_folder: str) -> None:
    """
    Copia arquivos de uma pasta de origem para uma pasta de destino.

    Args:
        source_folder (str): Caminho da pasta de origem.
        destination_folder (str): Caminho da pasta de destino.

    Returns:
        None: Apenas realiza a cópia dos arquivos.

    Exemplo:
        copy_files("../data/data_processing/data_M-C-A/valid", "../data/data_processing/data_M-C-A-E/valid")
        # Copia as imagens de validação para a pasta data_M-C-A-E.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for file in files:
        src_path = os.path.join(source_folder, file)
        dst_path = os.path.join(destination_folder, file)
        try:
            shutil.copy2(src_path, dst_path)
            print(f"Imagem copiada: {file} de {src_path} para {dst_path}")
        except Exception as e:
            print(f"Erro ao copiar {file}: {e}")


if __name__ == "__main__":
    process_data_folders()
