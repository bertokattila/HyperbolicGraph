//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
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
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
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


	vec2 points[numberOfPoints]; // graf pontjait tarolo lista

	vec3 hyperBolicPoints[numberOfPoints];

	struct PointPair { // elt reprezentalo struct, csak az egyszerubb hasznalat miatt
		int a;
		int b;
	};

	PointPair edges[numberOfEdges]; // pontok indexei, amik szomszedosak

	std::vector<vec2> edgeCoordinates; // az elek vegpontjainak koordinatai



public:
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

		refreshEdgeCoordinates(points, edgeCoordinates);

		numberOfIntersections(edgeCoordinates);

		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer object
		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
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
			sizeof(points),  // # bytes
			&points[0],	      	// address
			GL_STATIC_DRAW);	// we do not change later

		glPointSize(10.0);
		glDrawArrays(GL_POINTS, 0 /*startIdx*/, numberOfPoints /*# Elements*/);

		/// Innentol az elek rajzolasa

		glUniform3f(colorLocation, 1, 0, 0); // mas szinuek legyenek

		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * edgeCoordinates.size(),  // # bytes
			&edgeCoordinates[0],	      	// address
			GL_STATIC_DRAW);	// we do not change later


		glDrawArrays(GL_LINES, 0 /*startIdx*/, edgeCoordinates.size() /*# Elements*/);

		glutSwapBuffers(); // exchange buffers for double buffering
	}
	void generateNewCoordinates(int seed, vec2 destinationArray[]) {

		srand(seed); // muszaj frissiteni, kulonben nem mukodik a heurisztikus rendezo

		// graf pontjainak generalasa
		for (int i = 0; i < numberOfPoints; i++)
		{
			// Random koordinatak generalasa -1 es 1 koze
			float xCoordinate = (((float)rand() / (float)RAND_MAX) * 2) - 1.0f;
			float yCoordinate = (((float)rand() / (float)RAND_MAX) * 2) - 1.0f;
			destinationArray[i] = vec2(xCoordinate, yCoordinate);
		}

		for (int i = 0; i < numberOfPoints; i++)
		{
			printf("%f\n", destinationArray[i].x);
			printf("%f\n", destinationArray[i].y);
		}
	}
	void refreshEdgeCoordinates(vec2 pointsArray[], std::vector<vec2> &edgeCoordinates) {
		edgeCoordinates.clear();
		for (int i = 0; i < numberOfEdges; i++)
		{
			printf("%d\n", edges[i].a);
			printf("%d\n", edges[i].b);

			edgeCoordinates.push_back(vec2(pointsArray[edges[i].a].x, pointsArray[edges[i].a].y));
			edgeCoordinates.push_back(vec2(pointsArray[edges[i].b].x, pointsArray[edges[i].b].y));

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
				vec2 iV = edgeCoordinates[i] - edgeCoordinates[i + 1];; // iranyvektor
				vec2 iN = vec2(-iV.y, iV.x); // normalvektor

				vec2 scalarMult = (iN * (edgeCoordinates[j] - edgeCoordinates[i])) * (iN * (edgeCoordinates[j + 1] - edgeCoordinates[i]));

				vec2 jV = edgeCoordinates[j] - edgeCoordinates[j + 1];; // iranyvektor
				vec2 jN = vec2(-jV.y, jV.x); // normalvektor

				vec2 scalarMult2 = (jN * (edgeCoordinates[i] - edgeCoordinates[j])) * (jN * (edgeCoordinates[i + 1] - edgeCoordinates[j]));


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
			float x = points[i].x;
			float y = points[i].y;
			float w = sqrt((x*x) + (y*y) + 1);
			hyperBolicPoints[i] = vec3(x, y, w);
			printf("x %f y %f w %f\n", x, y, w);
		}
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
		return (5 / distance) * 0.5 * ((distance - 0.2) * (distance - 0.2) * (distance - 0.2));

	}
	float notPairForce(float distance){

	}
	void forceBasedArrange() { // minden pont tomege 1

		vec2 velocities[numberOfPoints];


		for each (vec2 v in velocities) // sebessegvektorok kinullazasa
		{
			v.x = 0;
			v.y = 0;
		}

		for (int i = 0; i < numberOfPoints; i++)
		{
			vec2 FPair = vec2(0,0);
			vec2 FNotPair = vec2(0,0);
			for (int j = 0; j < numberOfPoints; j++)
			{
				if (areNeighbours(i, j)) {
					float dist = length(points[j] - points[i]); // i-bol j-be mutato vektor
					printf("dist %f force %f\n ",dist, pairForce(dist));
					/// pozitiv erovel a szummaba
				}
				else {
					/// negativ erovel a szummaba
				}
			}
		}
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
void onDisplay() {
	
	graph.draw();
	
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
	printf("Pressed: %d", key); // 32 a space
	if (key == 32) {
		//graph.heuristicArrange();
		//graph.draw();
		graph.forceBasedArrange();
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char* buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
