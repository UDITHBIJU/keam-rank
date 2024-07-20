import  data from './combined_data.js'



function findNearestRanks(rank, caste,course) {

   const csabs = data[course];
	const filtered = csabs.filter((obj) => obj[caste] !== undefined);

	filtered.sort(
		(a, b) => Math.abs(a[caste] - rank) - Math.abs(b[caste] - rank)
	);

	const nearestRanks = filtered.slice(0, 5);

	return nearestRanks;
}

export default findNearestRanks;

