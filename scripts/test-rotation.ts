/**
 * Test rotation pipeline on RunPod
 *
 * Usage:
 *   npx tsx scripts/test-rotation.ts
 */

const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY;
const RUNPOD_ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID;
const BLOB_TOKEN = process.env.BLOB_READ_WRITE_TOKEN;

if (!RUNPOD_API_KEY || !RUNPOD_ENDPOINT_ID) {
	console.error('Missing RUNPOD_API_KEY or RUNPOD_ENDPOINT_ID');
	process.exit(1);
}

if (!BLOB_TOKEN) {
	console.error('Missing BLOB_READ_WRITE_TOKEN');
	process.exit(1);
}

async function testRotation() {
	const jobId = `test-rotation-${Date.now()}`;

	// Use a simple test image URL (white cube on transparent background)
	const testImageUrl =
		'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg';

	const payload = {
		input: {
			job_id: jobId,
			job_type: 'rotation',
			input_image_url: testImageUrl,
			elevation: 20,
			blob_token: BLOB_TOKEN,
		},
	};

	console.log('Submitting rotation job to RunPod...');
	console.log('Job ID:', jobId);
	console.log('Endpoint:', RUNPOD_ENDPOINT_ID);

	const response = await fetch(
		`https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync`,
		{
			method: 'POST',
			headers: {
				Authorization: `Bearer ${RUNPOD_API_KEY}`,
				'Content-Type': 'application/json',
			},
			body: JSON.stringify(payload),
		},
	);

	const result = await response.json();

	if (!response.ok) {
		console.error('❌ Failed to submit job');
		console.error('Status:', response.status);
		console.error('Response:', JSON.stringify(result, null, 2));
		process.exit(1);
	}

	console.log('\n✅ Job submitted successfully!');
	console.log('Response:', JSON.stringify(result, null, 2));

	if (result.status === 'COMPLETED') {
		console.log('\n🎉 Rotation completed!');
		if (result.output?.results) {
			console.log('\nGenerated rotation URLs:');
			for (const [key, url] of Object.entries(result.output.results)) {
				console.log(`  ${key}: ${url}`);
			}
		}
	} else if (result.status === 'FAILED') {
		console.error('\n❌ Job failed:', result.error);
	} else {
		console.log('\nJob status:', result.status);
		console.log('Check RunPod dashboard for progress.');
	}
}

testRotation().catch(console.error);
