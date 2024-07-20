import express from "express";
import "body-parser";
import findNearestRanks from "./a.js";
const app = express();
const PORT = process.env.PORT || 3000;
// Middleware
app.use(express.urlencoded({ extended: true }));
app.use(express.static("public"));
app.set("view engine", "ejs");

 
 
// Route to render index.ejs 
app.get("/", (req, res) => {
	res.render("index", { colleges: [],rank:'',caste:'',course:''}); // Initial render without colleges
});

app.post("/colleges", (req, res) => {
	const { rank, caste, course } = req.body;

  const colleges = findNearestRanks(rank, caste, course);

console.log(colleges);
	res.render("index", { colleges ,caste,rank,course}); // Render index.ejs with colleges data
});

// Start server
app.listen(PORT, () => {
	console.log(`Server is running on http://localhost:${PORT}`);
});


  