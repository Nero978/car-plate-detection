// src-electron/main.js
const { app, BrowserWindow } = require('electron')
const { join } = require('path')
const { exec } = require('child_process');

// 屏蔽安全警告
// ectron Security Warning (Insecure Content-Security-Policy)
process.env['ELECTRON_DISABLE_SECURITY_WARNINGS'] = 'true'

let flaskProcess;

// 创建浏览器窗口时，调用这个函数。
const createWindow = () => {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
  })

  win.loadURL('http://localhost:5000')
  // development模式
  if (process.env.VITE_DEV_SERVER_URL) {
    win.loadURL(process.env.VITE_DEV_SERVER_URL)
    // 开启调试台
    win.webContents.openDevTools()
  } else {
    win.loadFile(join(__dirname, '../dist/index.html'))
  }
}

// Electron 会在初始化后并准备
app.whenReady().then(() => {
  flaskProcess = exec('python backend/app.py', (error, stdout, stderr) => {
    if (error) {
      console.error(`Flask error: ${error}`);
      return;
    }
    console.log(`Flask stdout: ${stdout}`);
    console.error(`Flask stderr: ${stderr}`);
  });

  createWindow()
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('quit', () => {
  // Kill the Flask process when Electron quits
  if (flaskProcess) {
    flaskProcess.kill();
  }
});