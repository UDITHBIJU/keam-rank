import data from "./combined_data.js";

function sortByCaste(array, caste) {
	return array.sort((a, b) => a[caste] - b[caste]);
}

function findNearestRanks(rank, caste, course) {
	const csabs = data[course];
	const filtered = csabs.filter((obj) => obj[caste] !== undefined);

	filtered.sort(
		(a, b) => Math.abs(a[caste] - rank) - Math.abs(b[caste] - rank)
	);
	let nearestRanks;
	if (rank >= 40000) {
		nearestRanks = filtered.slice(0, 50);
	} else if (rank >= 30000) {
		nearestRanks = filtered.slice(0, 40);
	} else if (rank >= 20000) {
		nearestRanks = filtered.slice(0, 30);
	} else {
		nearestRanks = filtered.slice(0, 20);
	}

	const newArray = sortByCaste(nearestRanks, caste);

	return newArray;
}

export default findNearestRanks;
