import express from "express";
import findNearestRanks from "./nearestRank.js";
import mongoose from "mongoose";
const app = express();
const PORT = process.env.PORT || 3000;
// Middleware
app.use(express.urlencoded({ extended: true }));
app.use("/public", express.static("public"));
app.set("view engine", "ejs");
mongoose.connect(
	"mongodb+srv://admin:admin@cluster0.eqevsdm.mongodb.net/keam?retryWrites=true&w=majority&appName=Cluster0",
	{
		useNewUrlParser: true,
		useUnifiedTopology: true,
	}
);

const visitorSchema = new mongoose.Schema({
	count: {
		type: Number,
		default: 0,
	},
});

const Count = mongoose.model("Visitor", visitorSchema);

 let out = ''	; 
 
// Route to render index.ejs 
app.get("/", async (req, res) => {
	res.render("index", { colleges: [],rank:'',caste:'',course:'',out:''}); // Initial render without colleges
	let count = await Count.findOne();
	if (!count) {
		count = new Count();
	}
	count.count+=1;
	await count.save();	
});

app.post("/colleges", (req, res) => {
	const { rank, caste, course } = req.body;

  const colleges = findNearestRanks(rank, caste, course);

  colleges.length>0?out='':out='No colleges found';

	res.render("index", { colleges ,caste,rank,course,out}); // Render index.ejs with colleges data

});

// Start server
app.listen(PORT, () => {
	console.log(`Server is running on http://localhost:${PORT}`);
});
  