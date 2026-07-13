import pickle
import shutil
import os
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm
import time

def load_totalembed(dataset):
    if dataset == "ml-25m":
        embedpath = "./Embeddings/ml-25m"
        datapath = "./ml-25m/user_5k/recbole/user_5k/"
        folder = "user_5k_total_ml-25m_5_32_full"
    elif dataset == "amazon-CDs_and_Vinyl":
        embedpath = "./Embeddings/amazon/CDs_and_Vinyl"
        datapath = "./amazon/user_5k/CDs_and_Vinyl/recbole/user_5k/"
        folder = "user_5k_total_amazon_5_32_full"
    with open("{}/user_totalembed.pkl".format(embedpath), "rb") as f:
        totalembed = pickle.load(f)
    embed_str = ""
    for i in range(len(totalembed[0])):
        embed_str += str(totalembed[0][i])
        embed_str += ", "
    
    if not os.path.exists("{}/user_5k_total_baseprofile".format(datapath)):
        os.mkdir("{}/user_5k_total_baseprofile".format(datapath))
    for postfix in ["user", "item", "train.inter", "valid.inter", "test.inter", "itememb"]:
        shutil.copy(
            "{}/{}/{}.{}".format(datapath, folder, folder, postfix),
            "{}/user_5k_total_baseprofile/user_5k_total_baseprofile.{}".format(datapath, postfix)
        )
    
    with open("{}/{}/{}.user".format(datapath, folder, folder), "r") as f:
        lines = f.read().splitlines()
    userid_list = []
    for i in range(1, len(lines)):
        cur_userid, _ = lines[i].split("|")
        userid_list.append(cur_userid)
    with open("{}/user_5k_total_baseprofile/user_5k_total_baseprofile.useremb".format(datapath), "w") as f:
        f.write("uid:token|user_emb:float_seq\n")
        for i in range(len(userid_list)):
            f.write("{}|{}\n".format(userid_list[i], embed_str))
    # for i in range(10):
    #     print(totalembed[i][:10])

def generate_BGE_user_embed(dataset):
    if dataset == "ml-25m":
        # path = "./ml-25m_results/ml-25m_3_64_full/checkpoints"
        path = "./ml-25m_results/ml-25m_5_32_full/checkpoints"
    elif dataset == "amazon-CDs_and_Vinyl":
        # path = "./amazon_results/amazon_5_48_full/checkpoints"
        path = "./amazon_results/amazon_5_32_full/checkpoints"
    user_id_list, sentence_list = [], []
    
    with tqdm(total=5000, desc="Loading user profiles...") as pbar:
        for files in os.listdir(path):
            if files.endswith("final_profile.pkl"):
                user_id = files.split("_")[1]
                if dataset == "ml-25m":
                    user_id = str(int(float(user_id)))
                user_id_list.append(user_id)
                with open("{}/{}".format(path, files), "rb") as f:
                    profile_list = pickle.load(f)
                sentence = ""
                for i in range(len(profile_list)):
                    sentence += profile_list[i]
                    if i < len(profile_list) - 1:
                        sentence += ", "
                    else:
                        sentence += "."
                sentence_list.append(sentence)
                pbar.update()
    
    model = BGEM3FlagModel('./models/BGE', use_fp16=True)
    embedding_list = model.encode(sentence_list, batch_size=12, max_length=8192)['dense_vecs']
    print(embedding_list.shape)
    with tqdm(total=len(embedding_list), desc="Saving embeddings...") as pbar:
        with open("./BGE_embeds_{}".format(dataset), "w") as f:
            f.write("uid:token|user_emb:float_seq\n")
            for i in range(len(user_id_list)):
                f.write("{}|".format(user_id_list[i]))
                for j in range(len(embedding_list[i])):
                    f.write(str(embedding_list[i][j]))
                    f.write(", ")
                f.write("\n")
                pbar.update()

def generate_BGE_item_embed(dataset):
    if dataset == "ml-25m":
        # path = "./ml-25m_results/ml-25m_3_64_full/checkpoints"
        path = "./ml-25m_results/ml-25m_5_32_full/checkpoints"
        item_path = "./ml-25m/user_5k/u.item"
    elif dataset == "amazon-CDs_and_Vinyl":
        # path = "./amazon_results/amazon_5_48_full/checkpoints"
        path = "./amazon_results/amazon_5_32_full/checkpoints"
        item_path = "./amazon/user_5k/CDs_and_Vinyl/u.item"
    item_id_list, sentence_list = [], []
    with open(item_path, "r") as f:
        lines = f.read().splitlines()
    with tqdm(total=len(lines), desc="Processing item profiles...") as pbar:
        for i in range(len(lines)):
            item_id, item_title, item_category, item_description = lines[i].split("|")
            item_id_list.append(item_id)
            sentence = item_title + ". " + item_category + item_description
            sentence_list.append(sentence)
            pbar.update()
    
    model = BGEM3FlagModel('./models/BGE', use_fp16=True)
    embedding_list = model.encode(sentence_list, batch_size=12, max_length=8192)['dense_vecs']
    print(embedding_list.shape)
    with tqdm(total=len(embedding_list), desc="Saving embeddings...") as pbar:
        with open("./BGE_embeds_{}_new".format(dataset), "w") as f:
            f.write("iid:token|item_emb:float_seq\n")
            for i in range(len(item_id_list)):
                f.write("{}|".format(item_id_list[i]))
                for j in range(len(embedding_list[i])):
                    f.write(str(embedding_list[i][j]))
                    f.write(", ")
                f.write("\n")
                pbar.update()

def generate_BGE_recbole(dataset):
    if dataset == "ml-25m":
        path = "./ml-25m/user_5k/recbole/user_5k/user_5k_total_baseprofile"
        new_folder = "user_5k_total_JINA_ori"
        new_path = "./ml-25m/user_5k/recbole/user_5k/{}".format(new_folder)
    elif dataset == "amazon-CDs_and_Vinyl":
        path = "./amazon/user_5k/CDs_and_Vinyl/recbole/user_5k/user_5k_total_baseprofile"
        new_folder = "user_5k_total_JINA_new"
        new_path = "./amazon/user_5k/CDs_and_Vinyl/recbole/user_5k/{}".format(new_folder)
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    for postfix in ["user", "item", "train.inter", "valid.inter", "test.inter"]:
        shutil.copy(
            "{}/user_5k_total_baseprofile.{}".format(path, postfix),
            "{}/{}.{}".format(new_path, new_folder, postfix)
        )

def test():
    path = "./ml-25m_results/ml-25m_3_64_full/checkpoints/user_1.0_final_profile.pkl"
    with open(path, "rb") as f:
        temp = pickle.load(f)
    print(temp)

def saving_user_totalprofiles(dataset):
    if dataset == "ml-25m":
        # path = "./ml-25m_results/ml-25m_3_64_full/checkpoints"
        path = "./ml-25m_results/ml-25m_5_32_full/checkpoints"
    elif dataset == "amazon-CDs_and_Vinyl":
        # path = "./amazon_results/amazon_5_48_full/checkpoints"
        path = "./amazon_results/amazon_5_32_full/checkpoints"
    
    user_id_list, sentence_list = [], []
    with tqdm(total=5000, desc="Loading user profiles...") as pbar:
        for files in os.listdir(path):
            if files.endswith("final_profile.pkl"):
                user_id = files.split("_")[1]
                if dataset == "ml-25m":
                    user_id = str(int(float(user_id)))
                user_id_list.append(user_id)
                with open("{}/{}".format(path, files), "rb") as f:
                    profile_list = pickle.load(f)
                sentence = ""
                for i in range(len(profile_list)):
                    sentence += profile_list[i]
                    if i < len(profile_list) - 1:
                        sentence += ", "
                    else:
                        sentence += "."
                sentence_list.append(sentence)
                pbar.update()
    
    with open("./{}_userprofiles_new".format(dataset), "w") as f:
        with tqdm(total=len(user_id_list), desc="Saving user profiles...") as pbar:
            for i in range(len(user_id_list)):
                f.write("{}|{}\n".format(user_id_list[i], sentence_list[i]))
                pbar.update()

if __name__ == "__main__":
    # load_totalembed("amazon-CDs_and_Vinyl")
    # generate_BGE_embed("amazon-CDs_and_Vinyl")
    generate_BGE_recbole("ml-25m")
    # generate_BGE_item_embed("ml-25m")
    # saving_user_totalprofiles("amazon-CDs_and_Vinyl")

# curl https://api.jina.ai/v1/embeddings \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer [Your jina API key]" \
#   -d @- <<EOFEOF
#   {
#     "model": "jina-embeddings-v3",
#     "task": "text-matching",
#     "late_chunking": false,
#     "dimensions": 1024,
#     "embedding_type": "float",
#     "input": [
#         "Organic skincare for sensitive skin with aloe vera and chamomile: Imagine the soothing embrace of nature with our organic skincare range, crafted specifically for sensitive skin. Infused with the calming properties of aloe vera and chamomile, each product provides gentle nourishment and protection. Say goodbye to irritation and hello to a glowing, healthy complexion.",
#         "Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille: Erleben Sie die wohltuende Wirkung unserer Bio-Hautpflege, speziell für empfindliche Haut entwickelt. Mit den beruhigenden Eigenschaften von Aloe Vera und Kamille pflegen und schützen unsere Produkte Ihre Haut auf natürliche Weise. Verabschieden Sie sich von Hautirritationen und genießen Sie einen strahlenden Teint.",
#         "Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla: Descubre el poder de la naturaleza con nuestra línea de cuidado de la piel orgánico, diseñada especialmente para pieles sensibles. Enriquecidos con aloe vera y manzanilla, estos productos ofrecen una hidratación y protección suave. Despídete de las irritaciones y saluda a una piel radiante y saludable.",
#         "针对敏感肌专门设计的天然有机护肤产品：体验由芦荟和洋甘菊提取物带来的自然呵护。我们的护肤产品特别为敏感肌设计，温和滋润，保护您的肌肤不受刺激。让您的肌肤告别不适，迎来健康光彩。",
#         "新しいメイクのトレンドは鮮やかな色と革新的な技術に焦点を当てています: 今シーズンのメイクアップトレンドは、大胆な色彩と革新的な技術に注目しています。ネオンアイライナーからホログラフィックハイライターまで、クリエイティビティを解き放ち、毎回ユニークなルックを演出しましょう。"
#     ]
#   }
# EOFEOF
