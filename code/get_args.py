import json
import utils


class Args:
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.dataset = config["dataset"]
        self.save_path = config["save_path"]
        self.train_path = config["train_path"]
        self.dev_path = config["dev_path"]
        self.test_path = config["test_path"]
        self.checkpoint_path = config["checkpoint_path"]

        self.epochs = config["epochs"]
        self.episodes = config["episodes"]
        self.episodes_q = config["episodes_q"]
        self.numFreeze = config["numFreeze"]
        self.filevocab = config["filevocab"]
        self.fileModelConfig = config["fileModelConfig"]
        self.fileModel = config["fileModel"]


        self.gpu = config["gpu"]
        self.seed = config["seed"]
        self.hidden_size = config["hidden_size"]
        self.nway = config["nway"]
        self.kshot = config["kshot"]
        self.qshot = config["qshot"]
        self.text_max_len = config["text_max_len"]
        self.label_max_len = config["label_max_len"]
        self.beta = config["beta"]
        self.step = config["step"]
        self.random_len = config["random_len"]
        self.it_num = config["it_num"]
        self.temprature = config["temprature"]
        self.temprature1 = config["temprature1"]
        self.alpha = config["alpha"]
        self.gama = config["gama"]
        self.se_layer = config["se_layer"]
        self.q_qshot = config["q_qshot"]
        self.margin = config["margin"]

        self.prompt_len = config["prompt_len"]
        self.key_hidden_size = config["key_hidden_size"]
        self.pool_len = config["pool_len"]
        self.key_init_method = config["key_init_method"]


        self.weight_decay = config["weight_decay"]
        self.learning_rate = config["learning_rate"]
        self.warmup_steps = config["warmup_steps"]
        self.dropout = config["dropout"]

        self.optionn = "bert_layer"
        self.dataset_name = "none"

        self.optionn = "bert_layer"
        self.dataset_name = "none"
        self.gama1 = 1.0
        self.gama2 = 1.0

        self.logger_path = "./"
        self.logger = utils.get_logger("aaa", self.logger_path)
        self.logger.info("come on!")






        # for k, v in args.__dict__.items():
        #     if v is not None:
        #         self.__dict__[k] = v

    def __repr__(self):
        return "{}".format(self.__dict__.items())

    def show_self(self):
        print("train_path: {} -----".format(self.train_path))
        print("dev_path: {} -----".format(self.dev_path))
        print("test_path: {} -----".format(self.test_path))
        print("checkpoint_path: {} -----".format(self.checkpoint_path))

        print("epochs: {} -----".format(self.epochs))
        print("episodes: {} -----".format(self.episodes))
        print("numFreeze: {} -----".format(self.numFreeze))
        print("filevocab: {} -----".format(self.filevocab))
        print("fileModelConfig: {} -----".format(self.fileModelConfig))
        print("fileModel: {} -----".format(self.fileModel))

        print("gpu: {} -----".format(self.gpu))
        print("seed: {} -----".format(self.seed))
        print("hidden_size: {} -----".format(self.hidden_size))
        print("nway: {} -----".format(self.nway))
        print("kshot: {} -----".format(self.kshot))
        print("qshot: {} -----".format(self.qshot))

        print("text_max_len: {} -----".format(self.text_max_len))
        print("label_max_len: {} -----".format(self.label_max_len))
        # print("beta: {} -----".format(self.beta))
        # print("step: {} -----".format(self.step))
        # print("random_len: {} -----".format(self.random_len))
        # print("it_num: {} -----".format(self.it_num))
        print("key_hidden_size: {} -----".format(self.key_hidden_size))
        # print("temprature1: {} -----".format(self.temprature1))
        # print("alpha: {} -----".format(self.alpha))
        # print("gama: {} -----".format(self.gama))
        print("pool_len: {} -----".format(self.pool_len))
        print("key_init_method: {} -----".format(self.key_init_method))

        print("prompt_len: {} -----".format(self.prompt_len))
        print("weight_decay: {} -----".format(self.weight_decay))
        print("learning_rate: {} -----".format(self.learning_rate))
        print("warmup_steps: {} -----".format(self.warmup_steps))

    def write_self(self,path):
        with open(path, 'w') as json_file:
            data = {'train_path': self.train_path}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'dev_path': self.dev_path}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'test_path': self.test_path}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'checkpoint_path': self.checkpoint_path}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')

            data = {'epochs': self.epochs}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'episodes': self.episodes}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'episodes_q': self.episodes_q}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'numFreeze': self.numFreeze}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'filevocab': self.filevocab}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'fileModelConfig': self.fileModelConfig}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'fileModel': self.fileModel}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')

            data = {'gpu': self.gpu}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'seed': self.seed}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'hidden_size': self.hidden_size}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'nway': self.nway}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'kshot': self.kshot}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'qshot': self.qshot}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')

            data = {'text_max_len': self.text_max_len}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'label_max_len': self.label_max_len}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'beta': self.beta}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'step': self.step}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'random_len': self.random_len}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'it_num': self.it_num}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'temprature': self.temprature}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'temprature1': self.temprature1}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'alpha': self.alpha}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'gama': self.gama}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'se_layer': self.se_layer}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'q_qshot': self.q_qshot}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')

            data = {'prompt_len': self.prompt_len}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'key_hidden_size': self.key_hidden_size}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'pool_len': self.pool_len}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'key_init_method': self.key_init_method}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')

            data = {'weight_decay': self.weight_decay}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'learning_rate': self.learning_rate}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'warmup_steps': self.warmup_steps}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'dropout': self.dropout}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            data = {'optionn': self.optionn}
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')







