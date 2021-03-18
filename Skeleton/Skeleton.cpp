//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Bertok Attila
// Neptun : I7XH6P
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char* const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec3 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x/vp.z, vp.y/vp.z, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char* const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao;	   // virtual world on the GPU

const int numberOfPoints = 50;
const int numberOfEdges = 61; // az osszes lehetseges el 5%-a ~ (50 alatt a 2) * 0.05


class Graph {

public:
	vec2 points[numberOfPoints]; // graf pontjait tarolo lista
	vec3 hyperbolicPoints[numberOfPoints];
	struct PointPair { // elt reprezentalo struct, csak az egyszerubb hasznalat miatt
		int a;
		int b;
	};

	PointPair edges[numberOfEdges]; // pontok indexei, amik szomszedosak

	std::vector<vec2> edgeCoordinates; // az elek vegpontjainak koordinatai
	std::vector<vec3> hyperbolicEdgeCoordinates; // az elek vegpontjainak koordinatai



	void create() {

		generateNewCoordinates(1, points);

		// graf eleinek generalasa
		for (int i = 0; i < numberOfEdges; i++)
		{
			bool edgeAlreadyExists = true;

			int a;
			int b;

			while (edgeAlreadyExists)
			{
				// Random parok generalasa 0 es 49 koze
				a = rand() % numberOfPoints;
				b = rand() % numberOfPoints;

				if (a == b) continue;

				// le kell ellenorizni, hogy letezik-e mar az el
				// i db mar letezo el van, ezert addig kell futnia a ciklusnak
				edgeAlreadyExists = false;
				for (int j = 0; j < i; j++)
				{
					if (edges[i].a == a && edges[i].b == b || edges[i].a == b && edges[i].b == a) {
						edgeAlreadyExists = true;
						break;
					}
				}
			}
			
			edges[i].a = a;
			edges[i].b = b;
		}

		refreshHyperbolicPoints();

		refreshEdgeCoordinates(hyperbolicPoints, edgeCoordinates, hyperbolicEdgeCoordinates);


		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer object
		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			3, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed


		// create program for the GPU
		gpuProgram.create(vertexSource, fragmentSource, "outColor");
		
	}
	void draw() {
		glClearColor(0, 0, 0, 0);     // background color
		glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer\

		// Set color to (0, 1, 0) = green
		int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(colorLocation, 0, 1, 0); // 3 floats

		float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
								  0, 1, 0, 0,    // row-major!
								  0, 0, 1, 0,
								  0, 0, 0, 1 };

		int MVPLocation = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(MVPLocation, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location
		glBindVertexArray(vao);  // Draw call
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(hyperbolicPoints),  // # bytes
			&hyperbolicPoints[0],	      	// address
			GL_STATIC_DRAW);	// we do not change later
		glPointSize(10.0);
		glDrawArrays(GL_POINTS, 0 /*startIdx*/, numberOfPoints /*# Elements*/);

		/// Innentol az elek rajzolasa
		glUniform3f(colorLocation, 1, 0, 0); // mas szinuek legyenek
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec3) * hyperbolicEdgeCoordinates.size(),  // # bytes
			&hyperbolicEdgeCoordinates[0],	      	// address
			GL_STATIC_DRAW);	// we do not change later
		glDrawArrays(GL_LINES, 0 /*startIdx*/, hyperbolicEdgeCoordinates.size() /*# Elements*/);
		glutSwapBuffers(); // exchange buffers for double buffering
	}
	void generateNewCoordinates(int seed, vec2 destinationArray[]) {

		srand(seed); // muszaj frissiteni, kulonben nem mukodik a heurisztikus rendezo

		// graf pontjainak generalasa
		for (int i = 0; i < numberOfPoints; i++)
		{
			// Random koordinatak generalasa -1 es 1 koze
			float xCoordinate = 2 * ((((float)rand() / (float)RAND_MAX) * 2) - 1.0f);
			float yCoordinate = 2 * ((((float)rand() / (float)RAND_MAX) * 2) - 1.0f);
			destinationArray[i] = vec2(xCoordinate, yCoordinate);
		}

		for (int i = 0; i < numberOfPoints; i++)
		{
			//printf("%f\n", destinationArray[i].x);
			//printf("%f\n", destinationArray[i].y);
		}
	}
	void refreshEdgeCoordinates(vec3 hyperbolicPointsArray[], std::vector<vec2> &edgeCoordinates, std::vector<vec3>& HyperbolicEdgeCoordinates) {
		edgeCoordinates.clear();
		HyperbolicEdgeCoordinates.clear();
		for (int i = 0; i < numberOfEdges; i++)
		{
			//printf("%d\n", edges[i].a);
			//printf("%d\n", edges[i].b);

			edgeCoordinates.push_back(vec2(hyperbolicPointsArray[edges[i].a].x, hyperbolicPointsArray[edges[i].a].y));
			edgeCoordinates.push_back(vec2(hyperbolicPointsArray[edges[i].b].x, hyperbolicPointsArray[edges[i].b].y));

			HyperbolicEdgeCoordinates.push_back(vec3(hyperbolicPointsArray[edges[i].a].x, hyperbolicPointsArray[edges[i].a].y, hyperbolicPointsArray[edges[i].a].z));
			HyperbolicEdgeCoordinates.push_back(vec3(hyperbolicPointsArray[edges[i].b].x, hyperbolicPointsArray[edges[i].b].y, hyperbolicPointsArray[edges[i].b].z));

		}
	}
	void refreshEdgeCoordinates(vec2 points[], std::vector<vec2>& edgeCoordinates) {
		edgeCoordinates.clear();
		for (int i = 0; i < numberOfEdges; i++)
		{
			//printf("%d\n", edges[i].a);
			//printf("%d\n", edges[i].b);

			edgeCoordinates.push_back(vec2(points[edges[i].a].x, points[edges[i].a].y));
			edgeCoordinates.push_back(vec2(points[edges[i].b].x, points[edges[i].b].y));

		}
	}
	void refreshDescartesFromHyperbolic() {
		for (int i = 0; i < numberOfPoints; i++)
		{
			points[i].x = hyperbolicPoints[i].x;
			points[i].y = hyperbolicPoints[i].y;
		}
	}
	int numberOfIntersections(std::vector<vec2> edgeCoordinates) {

		// egymast koveto ketto koordinata reprezental egy szakaszt pl 0-1; 2-3 ...
		// az igy megtalalt metszesek szamat 2 vel kell osztani, hiszen az algoritmus 2x szamolja az osszeset
		int numberOfIntersections = 0;
		for (int i = 0; i < edgeCoordinates.size(); i+=2)
		{
			for (int j = 0; j < edgeCoordinates.size(); j += 2)
			{
				// az eloadason tanultak alapjan 
				vec3 iV = edgeCoordinates[i] - edgeCoordinates[i + 1];; // iranyvektor
				vec3 iN = vec2(-iV.y, iV.x); // normalvektor

				vec3 scalarMult = (iN * (edgeCoordinates[j] - edgeCoordinates[i])) * (iN * (edgeCoordinates[j + 1] - edgeCoordinates[i]));

				vec3 jV = edgeCoordinates[j] - edgeCoordinates[j + 1];; // iranyvektor
				vec3 jN = vec2(-jV.y, jV.x); // normalvektor

				vec3 scalarMult2 = (jN * (edgeCoordinates[i] - edgeCoordinates[j])) * (jN * (edgeCoordinates[i + 1] - edgeCoordinates[j]));


				if (scalarMult.x + scalarMult.y < 0 && scalarMult2.x + scalarMult2.y < 0) { // i szakasz vegpontjai az j egyenesenek kulonbozo oldalain vannak es forditva
					numberOfIntersections++;
				}
				
			}
		}
		numberOfIntersections = numberOfIntersections / 2;
		printf("intersections: %d\n", numberOfIntersections);
		return numberOfIntersections;
	}
	void refreshHyperbolicPoints() {
		for (int i = 0; i < numberOfPoints; i++)
		{
			hyperbolicPoints[i] = descartesToHyperbolic(points[i]);
		}
	}
	vec3 descartesToHyperbolic(vec2 descartes) {
		float x = descartes.x;
		float y = descartes.y;
		float w = sqrt((x * x) + (y * y) + 1);
		return vec3(x, y, w);
	}
	void heuristicArrange() {
		int i = 0;
		int bestNumberOfInteresections = numberOfIntersections(edgeCoordinates);
		//vec2 bestPoints[numberOfPoints];
		//std::vector<vec2> bestEdgeCoordinates;

		vec2 tmpPoints[numberOfPoints];
		std::vector<vec2> tmpEdgeCoordinates;
		for (int i = 0; i < 10; i++)
		{
			tmpEdgeCoordinates.clear();
			generateNewCoordinates(i, tmpPoints);
			refreshEdgeCoordinates(tmpPoints, tmpEdgeCoordinates);
			if (numberOfIntersections(tmpEdgeCoordinates) < bestNumberOfInteresections) {
				memcpy(points, tmpPoints, sizeof(points));
				bestNumberOfInteresections = numberOfIntersections(tmpEdgeCoordinates);
			}

			i++;
		}
		refreshEdgeCoordinates(points, edgeCoordinates);
		refreshHyperbolicPoints();
		refreshEdgeCoordinates(hyperbolicPoints, edgeCoordinates, hyperbolicEdgeCoordinates);

	}
	void heuristicArrange2() {// k means
		
		for (int i = 0; i < numberOfPoints; i++)
		{
			vec2 sum = vec2(0,0);
			for (int j = 0; j < numberOfPoints; j++)
			{
				if (i == j) continue; /// sajat maga nem szamit
				if (areNeighbours(i, j)) {
					sum = sum + points[j]; 
				}
				else
				{
					sum = sum - points[j];
				}
			}
			points[i] = sum / (numberOfPoints - 1);
		}
		
		refreshHyperbolicPoints();
		refreshEdgeCoordinates(hyperbolicPoints, edgeCoordinates, hyperbolicEdgeCoordinates);
	}
	bool areNeighbours(int i, int j) {
		for each (PointPair pair in edges)
		{
			if (pair.a == i && pair.b == j || pair.a == j && pair.b == i) return true;
		}
		return false;
	}
	float pairForce(float distance) {
		float optimalDistance = 0.2;
		return (5 / distance) * 0.5 * ((distance - 0.8) * (distance - 0.8) * (distance - 0.8));
	}
	float notPairForce(float distance){return 1 / (-1 * ((sqrt(distance) * 10) * ((sqrt(distance) * 10))));}
	float lorentz(vec3 a, vec3 b) { return (a.x * b.x + a.y * b.y - a.z * b.z); }
	float hyperbolicDistance(vec3 a, vec3 b) { return acoshf(-lorentz(a, b)); }
	void forceBasedArrange() { // minden pont tomege 1

		vec3 velocities[numberOfPoints];
		for each (vec3 v in velocities) // sebessegvektorok kinullazasa
		{
			v.x = 0;
			v.y = 0;
			v.z = 0;
		}
		float dt = 0.2;
		for (float t = 0; t < 1; t += dt) /// ido halad elore
		{
			printf("\n\nt: %f\n\n ", t);
			for (int i = 0; i < numberOfPoints; i++)
			{

				//if (fabs(t - 0.4) < t * .001){
					//printf("\ni : %d x %f y: %f z: %f ", i,  hyperbolicPoints[i].x, hyperbolicPoints[i].y, hyperbolicPoints[i].z);

				//}
				vec3 FSum = vec3(0, 0, 0);
				for (int j = 0; j < numberOfPoints; j++)
				{
					if (i == j) continue;
					float dist = hyperbolicDistance(hyperbolicPoints[i], hyperbolicPoints[j]);

					if (areNeighbours(i, j)) {
						float forceSize = pairForce(dist);
						vec3 forceDirection = (hyperbolicPoints[j] - hyperbolicPoints[i]) * coshf(dist) / sinhf(dist);
						FSum = FSum + forceDirection * forceSize;
							/*if (i == 11) {
								printf("\n");
								printf("\n FSumx: %f, FSumy: %f, FSumz: %f ", FSum.x, FSum.y, FSum.z);
								printf("\ni : %d j: %d dx %f fdy: %f fdz: %f dist: %f", i, j,  forceDirection.x, forceDirection.y, forceDirection.z, dist);
								printf("\ni : %d j: %d ix %f iy: %f iz: %f  jx %f jy: %f jz: %f dist: %f", i, j, hyperbolicPoints[i].x, hyperbolicPoints[i].y, hyperbolicPoints[i].z, hyperbolicPoints[j].x, hyperbolicPoints[j].y, hyperbolicPoints[j].z, dist);
							}*/
						}
					else {
						continue; // ideiglenes
						float forceSize = notPairForce(dist);
						vec3 forceDirection = (hyperbolicPoints[j] - hyperbolicPoints[i]) * coshf(dist) / sinhf(dist);
						FSum = FSum + forceDirection * forceSize;
						/*if (i == 11) {
							printf("\n");
							printf("\n FSumx: %f, FSumy: %f, FSumz: %f ", FSum.x, FSum.y, FSum.z);
							printf("\ni : %d j: %d dx %f fdy: %f fdz: %f dist: %f", i, j,  forceDirection.x, forceDirection.y, forceDirection.z, dist);
							printf("\ni : %d j: %d ix %f iy: %f iz: %f  jx %f jy: %f jz: %f dist: %f", i, j, hyperbolicPoints[i].x, hyperbolicPoints[i].y, hyperbolicPoints[i].z, hyperbolicPoints[j].x, hyperbolicPoints[j].y, hyperbolicPoints[j].z, dist);
						}*/
						//printf("\ni : %d j: %d ix %f iy: %f iz: %f  jx %f jy: %f jz: %f dist: %f", i, j, hyperbolicPoints[i].x, hyperbolicPoints[i].y, hyperbolicPoints[i].z, hyperbolicPoints[j].x, hyperbolicPoints[j].y, hyperbolicPoints[j].z, dist);
					}

					/*if (abs(FSum.x) < 0.002) FSum.x = 0;
					if (abs(FSum.y) < 0.002) FSum.y = 0;
					if (abs(FSum.z) < 0.002) FSum.z = 0;
					*/
				}

				//printf("\ni : %d x: %f y: %f z: %f", i, FSum.x, FSum.y, FSum.z);

				//printf("\n normalllt bef FSumx: %f, FSumy: %f, FSumz: %f ", FSum.x, FSum.y, FSum.z);

				if (abs(FSum.x * dt) < 0.002) {
					FSum.x = 0;
				}
				if (abs(FSum.y * dt) < 0.002) {
					FSum.y = 0;
				}
				if (abs(FSum.z * dt) < 0.002) {
					FSum.z = 0;
				}

				
				//printf("\n normalllt FSumx: %f, FSumy: %f, FSumz: %f ", FSum.x, FSum.y, FSum.z);
				
				
				//printf("\n bvelocities vx: %f, vy: %f, vz: %f ", velocities[i].x, velocities[i].y, velocities[i].z);
				// v = v + F * m, de m = 1
				// v = v + F
				velocities[i] = velocities[i] + FSum *dt;

				//printf("\n velocities vx: %f, vy: %f, vz: %f ", velocities[i].x, velocities[i].y, velocities[i].z);
				

				//printf("i: %d f: %f\n ", i,  length(FSum));

			}
			// mozgatas es a sebessegvektor az uj pont erintosikjaba allitasa
			for (int i = 0; i < numberOfPoints; i++)
			{
				printf("\n%d\n", i);
				// v * t = s
				float motionDistance = length(velocities[i]) * dt;
				if (abs(motionDistance) < 0.05) { continue;  }
				vec3 hyperbolicPointsTemp = hyperbolicPoints[i] * cosh(motionDistance) + normalize(velocities[i]) * sinh(motionDistance);
				if (abs(hyperbolicPointsTemp.x) < 0.002) {
					hyperbolicPointsTemp.x = 0;
				}
				if (abs(hyperbolicPointsTemp.y) < 0.002) {
					hyperbolicPointsTemp.y = 0;
				}
				if (abs(hyperbolicPointsTemp.z) < 0.002) {
					hyperbolicPointsTemp.z = 0;
				}
				//printf("\ni : %d x: %f y: %f z: %f", i, hyperbolicPointsTemp.x, hyperbolicPointsTemp.y, hyperbolicPointsTemp.z);
			
				// viszont most az sebessegvektor kimutatna a hiperbolikus sikbol, ezert generalok egy pontot a sebessegvektor egyenesen, de valamivel tavolabb
				vec3 anOtherPointThatDirection = hyperbolicPoints[i] * cosh(motionDistance + 2) + normalize(velocities[i]) * sinh(motionDistance + 2);
				// elvileg a tavolsaguk 1, ezert felesleges ujra kiszamolni TODO atirni
				printf("\n motionDist %f ", motionDistance);
				printf("\n point x: %f, y: %f, z: %f ", hyperbolicPoints[i].x, hyperbolicPoints[i].y, hyperbolicPoints[i].z);
				printf("\n newPoint x: %f,vy: %f, z: %f ", hyperbolicPointsTemp.x, hyperbolicPointsTemp.y, hyperbolicPointsTemp.z);
				printf("\n anOtherPointThatDirection vx: %f, vy: %f, vz: %f ", anOtherPointThatDirection.x, anOtherPointThatDirection.y, anOtherPointThatDirection.z);
				
				hyperbolicPoints[i] = hyperbolicPointsTemp;
				
				vec3 newVelocityVector = normalize((anOtherPointThatDirection - hyperbolicPoints[i] * cosh(hyperbolicDistance(hyperbolicPoints[i], anOtherPointThatDirection))) / sinh(hyperbolicDistance(hyperbolicPoints[i], anOtherPointThatDirection)));
				printf("\n 1legyen: %f ", length(normalize(newVelocityVector)));
				printf("\n newVelocityVector vx: %f, vy: %f, vz: %f ", newVelocityVector.x, newVelocityVector.y, newVelocityVector.z);
				velocities[i] = length(velocities[i]) * newVelocityVector;
			}


			refreshDescartesFromHyperbolic();
			refreshEdgeCoordinates(hyperbolicPoints, edgeCoordinates, hyperbolicEdgeCoordinates);
			draw();

		}
	}
	vec3 hyperbolicMirror(vec3 pointToMirror, vec3 mirrorPoint) {
		float distance = hyperbolicDistance(mirrorPoint, pointToMirror); // a tavolsag a pont es m1 kozott
		vec3 direction = (mirrorPoint - pointToMirror * coshf(distance)) / sinhf(distance); // az adott pontban ervenyes iranyvektor az m1 pont fele
		vec3 mirroredPoint = pointToMirror * coshf(2 * distance) + direction * sinhf(2 * distance);
		return mirroredPoint;
	}
	void move(vec3 hyperbolicMotionVector) {
		float hyperbolicMotionVectorLength = hyperbolicDistance(hyperbolicMotionVector, vec3(0, 0, 1)); // az eltolas hossza
		vec3 hyperbolicMotionVectorDirection = (hyperbolicMotionVector - vec3(0, 0, 1) * coshf(hyperbolicMotionVectorLength)) / sinhf(hyperbolicMotionVectorLength); /// a hiperbola origojaban ervenyes iranyvektor

		// ketto tukrozesi pont valasztasa az hiperbolikus szakasz egyenlete alapjan, fontos, hogy a ketto kozotti tavolsag fele legyen a kivant eltolas tavolsaganak
		vec3 m1 = vec3(0, 0, 1);
		vec3 m2 = vec3(0, 0, 1) * coshf(hyperbolicMotionVectorLength * 0.5) + hyperbolicMotionVectorDirection * sinhf(hyperbolicMotionVectorLength * 0.5);

		for (int i = 0; i < numberOfPoints; i++)
		{
			vec3 firstMirrored = hyperbolicMirror(hyperbolicPoints[i], m1); // pont tukrozese m1-re
			hyperbolicPoints[i] = hyperbolicMirror(firstMirrored, m2); // m1-re tukrozott pont tukrozese m2-re, az igy kapott pont lesz az uj pozicioja
		}
		refreshDescartesFromHyperbolic();
		refreshEdgeCoordinates(hyperbolicPoints, edgeCoordinates, hyperbolicEdgeCoordinates);
		draw();
	}
};

Graph graph;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	graph.create();
	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {graph.draw();}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
	printf("Pressed: %d", key); // 32 a space
	if (key == 32) {
		graph.heuristicArrange2();
		graph.draw();
		graph.forceBasedArrange();
	}
}

vec2 motionStartCoordinates, motionEndCoordinates;
// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	motionEndCoordinates = vec2(cX, cY); 
	vec2 descartesMotionVector = vec2((motionEndCoordinates - motionStartCoordinates).x, (motionEndCoordinates - motionStartCoordinates).y); // jelenlegi iteracioban az eger alapjan eltolasvektor
	if (abs(descartesMotionVector.x) <= 0.0005 || abs(descartesMotionVector.y) <= 0.0005) {// kis eltolasnal elszallnanak a pontok, szerintem azert mert pontatlan a float
		return;	//  ezert ilyen kis eltolast nem engedek meg, de ha tovabb huzza az egeret, akkor egyszerre, amikor mar eleg nagy az eltolas, meg fog tortenni
 	}
	graph.move(graph.descartesToHyperbolic(descartesMotionVector)); // az eltolasvektor hierbolikus megfelelojevel tortenik az eltolas
	motionStartCoordinates = motionEndCoordinates; // a kovetkezo lefutasnal a mar megtortent eltolast ne csinalja meg ujra
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	if (button == GLUT_LEFT_BUTTON) {
		motionStartCoordinates = vec2(cX, cY);
		printf("\nklikk %f %f", cX, cY);
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
