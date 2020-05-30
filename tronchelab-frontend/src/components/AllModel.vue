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
      <v-col class="mx-auto" cols="5">
        <v-card shaped color="purple darken-2" dark>
          <v-card-title class="justify-center headline"
            >All models</v-card-title
          >
          <v-card-subtitle>Send us the image you want to test</v-card-subtitle>
          <v-file-input
            class="ma-3"
            outlined
            chips
            label="File input"
            placeholder="Upload your image here"
            dark
          ></v-file-input>
        </v-card>
      </v-col>
      <v-col class="ma-auto" cols="5">
        <v-card shaped color="purple darken-2" dark>
          <v-card-title class="justify-center headline">
            Result will be here
          </v-card-title>
          <v-card-text class="white--text">
            <p class="text-base">According to the AI, the result is:</p>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
    <v-row>
      <v-col class="mx-auto" cols="5">
        <v-card shaped color="purple darken-3" dark>
          <v-card-title class="justify-center headline"
            >Linear model</v-card-title
          >
          <v-card-subtitle>Send us the image you want to test</v-card-subtitle>
          <v-file-input
            class="ma-3"
            outlined
            chips
            label="File input"
            placeholder="Upload your image here"
            dark
          ></v-file-input>
        </v-card>
      </v-col>
      <v-col class="ma-auto" cols="5">
        <v-card shaped color="purple darken-3" dark>
          <v-card-title class="justify-center headline">
            Result will be here
          </v-card-title>
          <v-card-text class="white--text">
            <p class="text-base">According to the AI, the result is:</p>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
    <v-row>
      <v-col class="mx-auto" cols="5">
        <v-card shaped color="deep-purple darken-1" dark>
          <v-card-title class="justify-center headline">CNN model</v-card-title>
          <v-card-subtitle>Send us the image you want to test</v-card-subtitle>
          <v-file-input
            class="ma-3"
            outlined
            chips
            label="File input"
            placeholder="Upload your image here"
            dark
          ></v-file-input>
        </v-card>
      </v-col>
      <v-col class="ma-auto" cols="5">
        <v-card shaped color="deep-purple darken-1" dark>
          <v-card-title class="justify-center headline">
            Result will be here
          </v-card-title>
          <v-card-text class="white--text">
            <p class="text-base">According to the AI, the result is:</p>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
    <v-row>
      <v-col class="mx-auto" cols="5">
        <v-card shaped color="deep-purple darken-2" dark>
          <v-card-title class="justify-center headline">MLP model</v-card-title>
          <v-card-subtitle>Send us the image you want to test</v-card-subtitle>
          <v-file-input
            class="ma-3"
            outlined
            chips
            label="File input"
            placeholder="Upload your image here"
            dark
          ></v-file-input>
        </v-card>
      </v-col>
      <v-col class="ma-auto" cols="5">
        <v-card shaped color="deep-purple darken-2" dark>
          <v-card-title class="justify-center headline">
            Result will be here
          </v-card-title>
          <v-card-text class="white--text">
            <p class="text-base">According to the AI, the result is:</p>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
    <v-row>
      <v-col class="mx-auto" cols="5">
        <v-card shaped color="purple darken-4" dark>
          <v-card-title class="justify-center headline"
            >Resnet model</v-card-title
          >
          <v-card-subtitle>Send us the image you want to test</v-card-subtitle>
          <v-file-input
            class="ma-3"
            outlined
            chips
            label="File input"
            placeholder="Upload your image here"
            dark
          ></v-file-input>
        </v-card>
      </v-col>
      <v-col class="ma-auto" cols="5">
        <v-card shaped color="purple darken-4" dark>
          <v-card-title class="justify-center headline">
            Result will be here
          </v-card-title>
          <v-card-text class="white--text">
            <p class="text-base">According to the AI, the result is:</p>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
export default {
  name: "AllModel",

  data: () => ({})
};
</script>

<style lang="scss" scoped>
.v-card--shaped {
  border-radius: 24px 4px !important;
}
</style>
