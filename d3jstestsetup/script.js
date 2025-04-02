// Set up dimensions
const margin = { top: 50, right: 50, bottom: 50, left: 100 };
const width = 1200 - margin.left - margin.right;
const height = 800 - margin.top - margin.bottom;

// Create an SVG container
const svg = d3.select("#graph-container")
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

// Load the JSON data
d3.json("../jsonexport/artist_collab.json").then(data => {
    const nodes = data.nodes;
    const links = data.links;

    // Create time scale (x-axis)
    const timeExtent = d3.extent(links.flatMap(d => d.release_dates.map(date => new Date(date))));
    const xScale = d3.scaleTime()
        .domain(timeExtent)
        .range([0, width]);

    // Create y-scale for artists
    const yScale = d3.scalePoint()
        .domain(nodes.map(d => d.id))  // Use artist IDs for positioning
        .range([0, height])
        .padding(0.5);

    // Draw x-axis
    svg.append("g")
        .attr("transform", `translate(0,${height})`)
        .call(d3.axisBottom(xScale).ticks(10));

    // Draw y-axis with artist names instead of IDs
    svg.append("g")
        .call(d3.axisLeft(yScale).tickFormat(d => nodes.find(n => n.id === d).name)) // Format y-axis ticks with artist names
        .selectAll(".tick text")  // Adjust the appearance of text labels
        .style("text-anchor", "start")
        .style("font-size", "2px");

    // Plot artist nodes
    svg.selectAll(".artist")
        .data(nodes)
        .enter()
        .append("circle")
        .attr("class", "artist")
        .attr("cx", 0)  // Start at the left
        .attr("cy", d => yScale(d.id))
        .attr("r", 0.5)
        .attr("fill", "steelblue");

    // Plot edges (collaborations) as horizontal lines
    svg.selectAll(".collaboration-line")
        .data(links)
        .enter()
        .append("line")
        .attr("class", "collaboration-line")
        .attr("x1", d => xScale(new Date(d.release_dates[0])))  // x position based on release date
        .attr("x2", d => xScale(new Date(d.release_dates[0])))  // x position for both ends of the line (horizontal)
        .attr("y1", d => yScale(d.source))  // y position based on the source artist
        .attr("y2", d => yScale(d.target))  // y position based on the target artist
        .attr("stroke", "gray")
        .attr("stroke-width", 1)
        .attr("stroke-opacity", 0.6);
});
