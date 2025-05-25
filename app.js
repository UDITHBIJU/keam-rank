const express = require("express");
const mongoose = require("mongoose");
const rout = require("./routes/routes");
const app = express();
const multer = require("multer");

app.use(express.urlencoded({ extended: false }));
app.use(express.json());

app.use("/public", express.static("public"));
app.set("view engine", "ejs");
app.use('/api',rout);
mongoose.connect(
	"mongodb+srv://admin:admin@cluster0.eqevsdm.mongodb.net/keam?retryWrites=true&w=majority&appName=Cluster0"
);



app.listen(4001);
