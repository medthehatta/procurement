module Main exposing (main)

import Browser
import Html exposing (Html, a, button, code, div, h1, p, text, img)
import Html.Attributes exposing (href, src, style)
import Html.Events exposing (onClick)
import VitePluginHelper


type Msg
    = Increment
    | Decrement


main : Program () Int Msg
main =
    Browser.sandbox { init = 0, update = update, view = view }


update : Msg -> number -> number
update msg model =
    case msg of
        Increment ->
            model + 1

        Decrement ->
            model - 1


view : Int -> Html Msg
view model =
    div []
        [ img [ src <| VitePluginHelper.asset "/src/assets/logo.png", style "width" "300px" ] []
        , helloWorld model
        ]


helloWorld : Int -> Html Msg
helloWorld model =
    div []
        [ h1 [] [ text "Hello, Vite + Elm!" ]
        , p []
            [ a [ href "https://vitejs.dev/guide/features.html" ] [ text "Vite Documentation" ]
            , text " | "
            , a [ href "https://guide.elm-lang.org/" ] [ text "Elm Documentation" ]
            ]
        , button [ onClick Increment ] [ text "+" ]
        , text <| "Count is: " ++ String.fromInt model
        , button [ onClick Decrement ] [ text "-" ]
        , p []
            [ text "Edit "
            , code [] [ text "src/Main.elm" ]
            , text " to test auto refresh"
            ]
        ]
