<template>
  <v-app>
    <v-app-bar app color="deep-purple darken-2" dark>
      <div class="d-flex align-center">
        <v-img
          alt="TroncheLab Logo"
          class="shrink mr-2"
          contain
          src="./assets/mask-emoji.png"
          transition="scale-transition"
          width="40"
        />
        <v-toolbar-title>Tronche Lab</v-toolbar-title>
      </div>

      <v-spacer></v-spacer>

      <v-btn href="https://github.com/Basti3n/TroncheLab" target="_blank" text>
        <span class="mr-2">Github Repository</span>
        <v-icon>mdi-open-in-new</v-icon>
      </v-btn>
    </v-app-bar>

    <v-content>
      <AllModel />
    </v-content>
  </v-app>
</template>

<script>
import AllModel from "./components/AllModel";

export default {
  name: "App",

  components: {
    AllModel
  },

  data: () => ({
    //
  })
};
</script>
