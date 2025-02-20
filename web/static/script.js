const init = (fen) => {
  const urlParams = new URLSearchParams(window.location.search);
  const rating = urlParams.get("rating") || 1200;

  var moves = [];
  var preds = [];
  const chart = new Chart("chart", {
    type: "line",
    data: {
      labels: moves,
      datasets: [
        {
          label: "Predictions",
          data: preds,
          borderColor: "rgb(255, 99, 132)",
          tension: 0.4,
        },
      ],
    },
  });

  var game = new Chess(fen);

  const onDragStart = (source, piece, position, orientation) => {
    if (game.game_over()) return false;
    if (game.turn() == "w" && piece.search(/^b/) !== -1) return false;
    if (game.turn() == "b" && piece.search(/^w/) !== -1) return false;
  };

  const onDrop = (source, target) => {
    const move = game.move({ from: source, to: target, promotion: "q" });
    if (move === null) return "snapback";
  };

  const onChange = async () => {
    moves.push(moves.length + 1);
    const fen = game.fen();
    var pred = await fetch(`/move/${fen}`);
    pred = await pred.text();
    preds.push(parseFloat(pred));
    chart.update();
    if (game.game_over()) alert("Game Over");
    if (game.turn() == "b") {
      const response = await fetch(`/maia/${game.fen()}?rating=${rating}`);
      const fen = await response.text();
      game.load(fen);
      board.position(game.fen());
    }
  };

  const config = {
    draggable: true,
    position: game.fen(),
    onDragStart,
    onDrop,
    onChange,
  };
  const board = Chessboard("board", config);

  $("#cheat").on("click", async () => {
    const response = await fetch(`/stockfish/${game.fen()}`);
    const fen = await response.text();
    game.load(fen);
    board.position(game.fen());
  });
};
