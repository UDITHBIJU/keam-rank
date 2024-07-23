import  data from './combined_data.js'

function sortByCaste(array, caste) {
	return array.sort((a, b) => a[caste] - b[caste]);
}


function findNearestRanks(rank, caste,course) {

   const csabs = data[course];
	const filtered = csabs.filter((obj) => obj[caste] !== undefined);

	filtered.sort(
		(a, b) => Math.abs(a[caste] - rank) - Math.abs(b[caste] - rank)
	);

	const nearestRanks = filtered.slice(0, 20);

	const newArray = sortByCaste(nearestRanks,caste)

	return newArray;
}

export default findNearestRanks;

