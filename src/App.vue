<template>
  <div class="bg"></div>
  <div class="page">
    <div class="container">
      <div class="card header">
        <div>
          <div class="title">
            {{ loading ? '处理中...' : '车牌识别' }}
          </div>
          <div>快速识别车牌号</div>
        </div>

        <div>
          <t-upload accept="video/*" :triggerButtonProps="{
            theme: 'primary',
            variant: 'base'
          }" :requestMethod="uploadVideo">
          </t-upload>
        </div>
      </div>

      <template v-if="result.video">
        <div class="card">
          <video class="media" :src="result.video" controls></video>
        </div>

        <div>
          <t-row :gutter="[32, 32]" align="stretch">
            <t-col :span="6">
              <div class="card">
                <img class="media" :src="result.frame" alt="">
              </div>
            </t-col>

            <t-col :span="6">
              <t-row :gutter="[32, 32]" align="stretch" class="plate-info">
                <t-col :span="6">
                  <div class="card">
                    <img class="media" :src="result.plate_img" alt="">
                  </div>
                </t-col>
                <t-col :span="6">
                  <div class="card plate-str">
                    {{ result.plate }}
                  </div>
                </t-col>
              </t-row>

              <div class="card plate-image">
                <template v-for="img in result.char_imgs">
                  <img :src="img" alt="">
                </template>
              </div>
            </t-col>
          </t-row>
        </div>
      </template>
    </div>
  </div>

</template>

<script setup>
import { ref } from 'vue';
import axios from 'axios';
import { MessagePlugin } from 'tdesign-vue-next';

const result = ref({
  video: '',
  frame: '',
  plate: '',
  plate_img: '',
  char_imgs: [],
});

const loading = ref(false);

function uploadVideo(file) {
  MessagePlugin.loading('识别中，请稍后...');
  loading.value = true;
  const formData = new FormData();
  formData.append('file', file.raw);
  console.log(file);
  return new Promise((resolve, reject) => {
    axios({
      method: 'post',
      url: 'http://localhost:3000/api/get-plate',
      data: formData,
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    }).then(res => {
      MessagePlugin.success('识别成功');
      console.log(res.data);
      result.value = res.data.data;
      result.value.video = 'http://localhost:3000/' + result.value.video;
      resolve({
        status: 'success'
      });
    }).catch(err => {
      MessagePlugin.error('识别失败');
      console.error(err);
      resolve({
        status: 'fail',
        error: '识别失败',
      });
    }).finally(() => {
      loading.value = false;
    });
  });
}

</script>

<style scoped>
.bg {
  background: url('./assets/bg.jpg') no-repeat;
  background-size: cover;
  background-attachment: fixed;
  background-position: center;
  height: 100vh;
  width: 100vw;
  position: fixed;
  top: 0;
  left: 0;
  z-index: -1;
}

.page {
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
}

.container {
  max-width: 1200px;
  width: 100%;
  height: 100%;
  margin: 0 2rem;
  padding: 2rem 0;
  box-sizing: border-box;
}

.card {
  width: 100%;
  height: 100%;
  background-color: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(10px);
  border-radius: 10px;
  padding: 2rem;
  box-sizing: border-box;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
  margin-bottom: 2rem;
}

.card:last-child {
  margin-bottom: 0;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header .title {
  font-size: 1.5rem;
  font-weight: bold;
  margin-top: 0;
  margin-bottom: 1rem;
}

.header .title+div {
  color: var(--td-text-color-secondary);
}

.table {
  width: 100%;
  height: 100%;
}

.plate-info {
  margin-bottom: 2rem;
}

.media {
  width: 100%;
  height: auto;
}

.plate-str {
  font: initial;
  font-size: 1.5rem;
  font-weight: bold;
  text-align: center;
}

.plate-image {
  height: auto;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
}

.plate-image img {
  width: 100%;
  height: auto;
}
</style>
