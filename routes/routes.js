const express = require('express');
const multer = require('multer');

const router = express.Router();
const {signin,admin, dashboard, upload, index,colleges} = require('../controllers/auth')
router.post('/signin',signin)
router.get('/singin',admin)
router.get('/dashboard',dashboard)
router.get('/',index)
router.post('/colleges',colleges)

const storage = multer.diskStorage({
    destination:'uploads/',
    filename:(req,file,cb)=>{
        cb(null,file.originalname)
    }
})
const uploadc = multer({storage:storage})
router.post('/upload',uploadc.single('file'),upload)
module.exports = router;    