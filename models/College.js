const mongoose = require("mongoose");

const CollegeSchema = new mongoose.Schema({
	ccode: { type: String, required: true }, // "Name of College"
	Type: { type: String, required: true }, // "Type"
	SM: { type: String, default: "-" }, // "SM"
	EW: { type: String, default: "-" },
	EZ: { type: String, default: "-" },
	MU: { type: String, default: "-" },
	BH: { type: String, default: "-" },
	LA: { type: String, default: "-" },
	DV: { type: String, default: "-" },
	VK: { type: String, default: "-" },
	BX: { type: String, default: "-" },
	KU: { type: String, default: "-" },
	KN: { type: String, default: "-" },
	SC: { type: String, default: "-" },
	ST: { type: String, default: "-" },
	courseName: { type: String, required: true }, // "courseName"
	cname: { type: String, }, // "cname"
});

module.exports = mongoose.model("college", CollegeSchema);
