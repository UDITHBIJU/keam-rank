const { name } = require("ejs");
const multer = require("multer");
const axios = require("axios");
const FormData = require("form-data");
const fs = require("fs");
const Colleges = require("../models/College"); // Import Model
const { type } = require("os");
const uload = multer({ dest: "uploads/" });

const signin = (req, res) => {
	const { username, password } = req.body;
	if (username === "admin" && password === "admin") {
		res.redirect("/api/dashboard");
	} else {
		res.send("Email and password are required");
	}
};
const admin = (req, res) => {
	res.render("index.ejs");
};
const names = [
	"Artificial Intelligence and Data Science",
	"Applied Electronics & Instrumentation",
	"B.Tech. (Agricultural Engg.)",
	"Artificial Intelligence and Machine Learning",
	"Artificial Intelligence",
	"B.Tech Agriculture Engineering",
	"Aeronautical Engineering",
	"Automobile Engineering",
	"Bio Technology and Biochemical Engg",
	"Computer Science and Engineering and Business Systems",
	"Computer Science and Engineering (Block Chain)",
	"Bio Medical Engineering",
	"Bio Technology",
	"Civil Engineering",
	"Computer Science & Design",
	"Chemical Engineering",
	"Computer Science & Engg. (Artificial Intelligence & Machine Learning)",
	"Computer Science & Engineering (Data Science)",
	"Computer Science & Engineering",
	"Computer Science & Engineering (Artificial Intelligence)",
	"Computer Science and Business Systems",
	"Civil and Environmental Engineering",
	"Computer Science and Engineering (Cyber Security)",
	"Dairy Technology",
	"Elcetronics and Communication (Advanced Communication Technology)",
	"Electronics & Biomedical Engineering",
	"Electronics & Communication Engineering",
	"Electrical & Electronics Engineering",
	"Electronics & Instrumentation",
	"Electrical and Computer Engineering",
	"Electronics and Computer Engineering",
	"Electronics Engineering (VLSI Design & Technology)",
	"Safety & Fire Engineering",
	"Food Technology",
	"Computer Science and Engineering (Internet of Things and Cyber Security including Block Chain Techno",
	"Instrumentation & Control Engg",
	"Computer Science & Engineering (Artificial Intelligence and Data Science)",
	"Industrial Engineering",
	"Computer Science and Engineering (Internet of Things)",
	"Information Technology",
	"Mechanical Engg. (Automobile)",
	"Mechanical Engineering",
	"Metallurgical and Materials Engineering",
	"Mechatronics Engineering",
	"Production Engineering",
	"Polymer Engineering",
	"Printing Technology",
	"Robotics and Artificial Intelligence",
	"Robotics & Automation",
	"Naval Arch. & Ship Building",
	"Architecture",
];
const dashboard = (req, res) => {
	
	res.render("dashboard.ejs", { names });
};

const upload = async (req, res) => {
	const username = req.body.username; // Get the selected name
	const file = req.file; // Get the uploaded file
	console.log(username);
	if (!file) {
		return res.status(400).send("No file uploaded.");
	}
	try {
		// Create a FormData object
		const formData = new FormData();
		formData.append("file", fs.createReadStream(file.path), file.originalname);

		// Send the file to the FastAPI server
		const response = await axios.post(
			"http://127.0.0.1:8000/extract",
			formData,
			{
				headers: {
					...formData.getHeaders(),
				},
			}
		);


		// Ensure response.data is an array before mapping
		const modifiedData = response.data.data.map((obj) => {
			const newObj = { ...obj, courseName: username }; // Add courseName

			// Rename "null" key to "cname" if it exists
			if (newObj.null !== undefined) {
				newObj.cname = newObj.null; // Copy value to "cname"
				delete newObj.null; // Remove the old "null" key
			}
			if (newObj["Name of College"] !== undefined) {
				newObj.ccode = newObj["Name of College"];
				delete newObj["Name of College"];
			}

			return newObj;
		});
		await Colleges.insertMany(modifiedData);
	res.render("dashboard.ejs", { names });
	} catch (error) {
		console.error("Error forwarding file:", error);
		res.status(500).send("Error processing the file.");
	}

	// res.send("File uploaded successfully.");
};
const colleges = async  (req, res) => {
	const { course,category,rank,ctype} = req.body;

    const response = await axios.post("http://localhost:8000/predict", {
			course,
			category,
			rank,
			ctype,
		});
  const colleges = response.data.recommended_colleges || [];

	colleges.length > 0 ? (out = "") : (out = "No colleges found"); 

	res.render("college", { colleges, category, rank, course,ctype, out ,names}); // Render index.ejs with colleges data
};

const index = async (req, res) => {
	res.render("college", {
		names, 
		colleges: [],
		rank: "",
		ctype: "",
		course: "",
		category: "",
		out: "",
	}); // Initial render without colleges
}
 
module.exports = { signin, admin, dashboard, upload, colleges,index };
