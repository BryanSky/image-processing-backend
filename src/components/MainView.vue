/* eslint-disable */
<template>
  <div class="main-view">
    <b-container>
      <b-row>
        <b-form-file
          v-model="file1"
          :state="Boolean(file1)"
          placeholder="Choose a file or drop it here..."
          drop-placeholder="Drop file here...">
        </b-form-file>
        <button v-on:click="submitFile()" class="mt-3" :disabled="file == null" >Upload {{ file ? file.name : '' }}</button>
      </b-row>
      <b-row>
        <div id="preview">
          <b-img></b-img>
        </div>
      </b-row>
      <b-row>
          <b-tabs content-class="mt-3">
            <b-tab title="JSON" active><p>I'm the first tab</p></b-tab>
            <b-tab title="GraphViz"><p>I'm the second tab</p></b-tab>
          </b-tabs>
      </b-row>
    </b-container>
  </div>
</template>

<script type="application/javascript">
import axios from 'axios'
export default {
  name: 'MainView',
  props: {
  },
  data: () => {
    return {
      file: null
    }
  },
  methods: {
    submitFile: () => {
      let formData = new FormData()
      formData.append('file', this.file)
      axios.post( '/upload',
        formData,
        {
          headers: {
              'Content-Type': 'multipart/form-data'
          }
        }
      ).then(function(response){
        console.log('SUCCESS!!')
        console.log(response)
      })
      .catch(function(response){
        console.log('FAILURE!!')
        console.log(response)
      });
    }
  }
}
</script>

<style scoped>
#preview {
  width:  80 %;
  height: 20 %;
}
</style>
