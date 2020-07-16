<template>
  <v-container>
    <v-row class="text-center">
      <v-col class="mb-4">
        <h1 class="display-2 font-weight-bold mb-3">
          Welcome to TroncheLab
        </h1>
      </v-col>
    </v-row>
    <v-row>
      <v-col class="ma-auto" cols="5">
        <v-card shaped color="purple darken-2" dark>
          <v-card-title class="justify-center headline"
            >All models</v-card-title
          >
          <v-card-subtitle>Send us the image you want to test</v-card-subtitle>
          <h5 class="mx-4">
            Paramètres utilisés : batch_size=42 | epoch=42 | accuracy=0.99
          </h5>
          <form enctype="multipart/form-data" class="flex flex-column">
            <v-file-input
              class="mx-3 mt-3"
              outlined
              chips
              label="File input"
              placeholder="Upload your image here"
              accept="image/*"
              @change="handleUpdateFile"
              dark
            ></v-file-input>
            <v-btn
              color="primary"
              @click="analyse('all')"
              class="mx-auto mb-2"
              dark
            >
              send
            </v-btn>
          </form>
        </v-card>
      </v-col>
      <v-col class="ma-auto" cols="5">
        <v-card shaped color="purple darken-2" dark>
          <v-card-title class="justify-center headline">
            Result will be here
          </v-card-title>
          <v-card-text class="white--text">
            <p class="text-base">
              According to the AI, the result is:
            </p>
            <p class="text-base">Linear model: {{ aiResult.linear }}</p>
            <p class="text-base">CNN model: {{ aiResult.cnn }}</p>
            <p class="text-base">MLP model: {{ aiResult.mlp }}</p>
            <p class="text-base">Resnet model: {{ aiResult.resnet }}</p>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
    <v-row>
      <v-col class="ma-auto" cols="5">
        <v-card shaped color="purple darken-3" dark>
          <v-card-title class="justify-center headline"
            >Linear models</v-card-title
          >
          <v-card-subtitle>Send us the image you want to test</v-card-subtitle>
          <h5 class="mx-4">
            Paramètres utilisés : batch_size=64 | epoch=1200 | accuracy=0.44
          </h5>
          <form enctype="multipart/form-data" class="flex flex-column">
            <v-file-input
              class="mx-3 mt-3"
              outlined
              chips
              label="File input"
              placeholder="Upload your image here"
              accept="image/*"
              @change="handleUpdateFile"
              dark
            ></v-file-input>
            <v-btn
              color="primary"
              @click="analyse('linear')"
              class="mx-auto mb-2"
              dark
            >
              send
            </v-btn>
          </form>
        </v-card>
      </v-col>
      <v-col class="ma-auto" cols="5">
        <v-card shaped color="purple darken-3" dark>
          <v-card-title class="justify-center headline">
            Result will be here
          </v-card-title>
          <v-card-text class="white--text">
            <p class="text-base">According to the AI, the result is:</p>
            <p class="text-base">Linear model: {{ aiResult.linear }}</p>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
    <v-row>
      <v-col class="mx-auto" cols="5">
        <v-card shaped color="deep-purple darken-1" dark>
          <v-card-title class="justify-center headline">CNN model</v-card-title>
          <v-card-subtitle>Send us the image you want to test</v-card-subtitle>
          <h5 class="mx-4">
            Paramètres utilisés : batch_size=100 | epoch=800 | accuracy=0.99
          </h5>
          <form enctype="multipart/form-data" class="flex flex-column">
            <v-file-input
              class="mx-3 mt-3"
              outlined
              chips
              label="File input"
              placeholder="Upload your image here"
              accept="image/*"
              @change="handleUpdateFile"
              dark
            ></v-file-input>
            <v-btn
              color="primary"
              @click="analyse('cnn')"
              class="mx-auto mb-2"
              dark
            >
              send
            </v-btn>
          </form>
        </v-card>
      </v-col>
      <v-col class="ma-auto" cols="5">
        <v-card shaped color="deep-purple darken-1" dark>
          <v-card-title class="justify-center headline">
            Result will be here
          </v-card-title>
          <v-card-text class="white--text">
            <p class="text-base">According to the AI, the result is:</p>
            <p class="text-base">CNN model: {{ aiResult.cnn }}</p>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
    <v-row>
      <v-col class="mx-auto" cols="5">
        <v-card shaped color="deep-purple darken-2" dark>
          <v-card-title class="justify-center headline">MLP model</v-card-title>
          <v-card-subtitle>Send us the image you want to test</v-card-subtitle>
          <h5 class="mx-4">
            Paramètres utilisés : batch_size=100 | epoch=1250 | accuracy=0.99
          </h5>
          <form enctype="multipart/form-data" class="flex flex-column">
            <v-file-input
              class="mx-3 mt-3"
              outlined
              chips
              label="File input"
              placeholder="Upload your image here"
              accept="image/*"
              @change="handleUpdateFile"
              dark
            ></v-file-input>
            <v-btn
              color="primary"
              @click="analyse('mlp')"
              class="mx-auto mb-2"
              dark
            >
              send
            </v-btn>
          </form>
        </v-card>
      </v-col>
      <v-col class="ma-auto" cols="5">
        <v-card shaped color="deep-purple darken-2" dark>
          <v-card-title class="justify-center headline">
            Result will be here
          </v-card-title>
          <v-card-text class="white--text">
            <p class="text-base">According to the AI, the result is:</p>
            <p class="text-base">MLP model: {{ aiResult.mlp }}</p>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
    <v-row>
      <v-col class="mx-auto" cols="5">
        <v-card shaped color="deep-purple darken-3" dark>
          <v-card-title class="justify-center headline"
            >Resnet model</v-card-title
          >
          <v-card-subtitle>Send us the image you want to test</v-card-subtitle>
          <h5 class="mx-4">
            Paramètres utilisés : batch_size=100 | epoch=1600 | accuracy=0.99
          </h5>
          <form enctype="multipart/form-data" class="flex flex-column">
            <v-file-input
              class="mx-3 mt-3"
              outlined
              chips
              label="File input"
              placeholder="Upload your image here"
              accept="image/*"
              @change="handleUpdateFile"
              dark
            ></v-file-input>
            <v-btn
              color="primary"
              @click="analyse('resnet')"
              class="mx-auto mb-2"
              dark
            >
              send
            </v-btn>
          </form>
        </v-card>
      </v-col>
      <v-col class="ma-auto" cols="5">
        <v-card shaped color="deep-purple darken-3" dark>
          <v-card-title class="justify-center headline">
            Result will be here
          </v-card-title>
          <v-card-text class="white--text">
            <p class="text-base">According to the AI, the result is:</p>
            <p class="text-base">Resnet model: {{ aiResult.resnet }}</p>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
    <v-row>
      <v-col class="mx-auto" cols="5">
        <v-card shaped color="purple darken-4" dark>
          <v-card-title class="justify-center headline"
            >Custom model</v-card-title
          >
          <v-card-subtitle>Send us the image you want to test</v-card-subtitle>
          <h5 class="mx-4">
            Paramètres utilisés : batch_size=42 | epoch=42 | accuracy=0.99
          </h5>
          <form enctype="multipart/form-data" class="flex flex-column">
            <div class="flex">
              <v-file-input
                class="mx-3 mt-3"
                outlined
                chips
                label="File input"
                placeholder="Upload your image here"
                accept="image/*"
                @change="handleUpdateFile"
                dark
              ></v-file-input>
              <v-text-field
                class="mx-3 mt-3"
                label="File name"
                placeholder="MyModel"
                v-model="customFile"
                outlined
              ></v-text-field>
            </div>
            <v-btn
              color="primary"
              @click="customAnalyse()"
              class="mx-auto mb-2"
              dark
            >
              send
            </v-btn>
          </form>
        </v-card>
      </v-col>
      <v-col class="ma-auto" cols="5">
        <v-card shaped color="purple darken-4" dark>
          <v-card-title class="justify-center headline">
            Result will be here
          </v-card-title>
          <v-card-text class="white--text">
            <p class="text-base">According to the AI, the result is:</p>
            <p class="text-base">{{ customFile }}: {{ aiResult.model }}</p>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
import axios from "axios";

export default {
  name: "AllModel",

  data() {
    return {
      apiUrl: "http://localhost:5000/api/",
      aiResult: "",
      image: null,
      customFile: ""
    };
  },
  methods: {
    handleUpdateFile(image) {
      this.image = image;
    },
    analyse(method) {
      let endpoint = "";
      let formData = new FormData();
      formData.append("image", this.image);
      switch (method) {
        case "all":
          endpoint = `${this.apiUrl}all`;
          break;
        case "linear":
          endpoint = `${this.apiUrl}linear`;
          break;
        case "cnn":
          endpoint = `${this.apiUrl}cnn`;
          break;
        case "mlp":
          endpoint = `${this.apiUrl}mlp`;
          break;
        case "resnet":
          endpoint = `${this.apiUrl}resnet`;
          break;
        default:
          break;
      }
      axios
        .post(endpoint, formData, {
          headers: {
            "Content-Type": "multipart/form-data"
          }
        })
        .then(res => {
          this.aiResult = res.data;
          alert("well analysed!");
        })
        .catch(err => console.log(err));
    },
    customAnalyse() {
      const endpoint = `${this.apiUrl}custom?path=${this.customFile}`;
      const formData = new FormData();
      formData.append("image", this.image);
      axios
        .post(endpoint, formData, {
          headers: {
            "Content-Type": "multipart/form-data"
          }
        })
        .then(res => {
          this.aiResult = res.data;
          alert("well analysed!");
        })
        .catch(err => console.log(err));
    }
  }
};
</script>

<style lang="scss">
.v-card--shaped {
  border-radius: 24px 4px !important;
}

.v-text-field__details {
  display: none !important;
}
</style>
